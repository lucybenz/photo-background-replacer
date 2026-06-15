using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using CvSize = OpenCvSharp.Size;

namespace PhotoBackgroundReplacer;

internal static class Program
{
    private static int Main(string[] args)
    {
        try
        {
            var options = Options.Parse(args);
            if (options.ShowHelp)
            {
                Console.WriteLine(Options.HelpText);
                return 0;
            }

            Directory.CreateDirectory(Path.GetDirectoryName(options.OutputPath)!);
            var person = Cv2.ImRead(options.InputPath, ImreadModes.Color);
            var background = Cv2.ImRead(options.BackgroundPath, ImreadModes.Color);
            if (person.Empty())
            {
                throw new FileNotFoundException("Input person image cannot be read.", options.InputPath);
            }

            if (background.Empty())
            {
                throw new FileNotFoundException("Background image cannot be read.", options.BackgroundPath);
            }

            using (person)
            using (background)
            {
                using var bg = CoverResize(background, person.Width, person.Height);
                using var result = RunMattingWithFallback(person, bg, options);
                Cv2.ImWrite(options.OutputPath, result);
            }

            Console.WriteLine("Saved result:");
            Console.WriteLine(options.OutputPath);
            TryOpenExplorer(options.OutputPath);
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
    }

    private static Mat RunMattingWithFallback(Mat person, Mat background, Options options)
    {
        var errors = new List<string>();
        foreach (var useGpu in options.Provider switch
        {
            ProviderMode.DirectML => new[] { true },
            ProviderMode.Cpu => new[] { false },
            _ => new[] { true, false }
        })
        {
            try
            {
                using var model = new RvmOnnxMatting(options.ModelPath, useGpu);
                Console.WriteLine($"Running RVM with {model.ProviderName}...");
                return model.Matte(person, options.DownsampleRatio, background);
            }
            catch (Exception ex)
            {
                errors.Add($"{(useGpu ? "DirectML" : "CPU")}: {ex.Message}");
            }
        }

        throw new InvalidOperationException("Matting failed. " + string.Join(" | ", errors));
    }

    private static Mat CoverResize(Mat src, int targetWidth, int targetHeight)
    {
        var scale = Math.Max(targetWidth / (double)src.Width, targetHeight / (double)src.Height);
        using var resized = new Mat();
        Cv2.Resize(src, resized, new CvSize((int)Math.Ceiling(src.Width * scale), (int)Math.Ceiling(src.Height * scale)), 0, 0, InterpolationFlags.Linear);
        var x = Math.Max(0, (resized.Width - targetWidth) / 2);
        var y = Math.Max(0, (resized.Height - targetHeight) / 2);
        var roi = new Rect(x, y, targetWidth, targetHeight);
        return new Mat(resized, roi).Clone();
    }

    private static void TryOpenExplorer(string selectedFile)
    {
        try
        {
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = $"/select,\"{selectedFile}\"",
                UseShellExecute = true
            });
        }
        catch
        {
            // Convenience only.
        }
    }
}

internal sealed class Options
{
    public string InputPath { get; private init; } = "";
    public string BackgroundPath { get; private init; } = "";
    public string OutputPath { get; private init; } = "";
    public string ModelPath { get; private init; } = "";
    public float DownsampleRatio { get; private init; } = 0.125f;
    public ProviderMode Provider { get; private init; } = ProviderMode.Auto;
    public bool ShowHelp { get; private init; }

    public static string HelpText =>
        """
        PhotoBackgroundReplacer

        Required:
          --input <person image>
          --background <background image>

        Optional:
          --output <result png>
          --model <rvm onnx model>
          --downsample <0.05-1.0>   default 0.125 for 4K quality
          --provider auto|dml|cpu   default auto

        Example:
          dotnet run --project photo_background_replacer\PhotoBackgroundReplacer -- --input person.jpg --background bg.jpg
        """;

    public static Options Parse(string[] args)
    {
        if (args.Length == 0 || args.Contains("--help") || args.Contains("-h"))
        {
            return new Options { ShowHelp = true };
        }

        var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length; i++)
        {
            var key = args[i];
            if (!key.StartsWith("--", StringComparison.Ordinal))
            {
                continue;
            }

            if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
            {
                map[key] = "true";
            }
            else
            {
                map[key] = args[++i];
            }
        }

        var root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
        var input = Require(map, "--input");
        var background = Require(map, "--background");
        var output = map.TryGetValue("--output", out var outPath)
            ? outPath
            : Path.Combine(root, "outputs", $"photo_bg_result_{DateTime.Now:yyyyMMdd_HHmmss}.png");
        var model = map.TryGetValue("--model", out var modelPath)
            ? modelPath
            : FindDefaultModel(root);
        var ratio = map.TryGetValue("--downsample", out var ratioText) && float.TryParse(ratioText, out var parsed)
            ? Math.Clamp(parsed, 0.05f, 1f)
            : 0.125f;
        var provider = map.TryGetValue("--provider", out var providerText)
            ? providerText.ToLowerInvariant() switch
            {
                "dml" or "directml" or "gpu" => ProviderMode.DirectML,
                "cpu" => ProviderMode.Cpu,
                _ => ProviderMode.Auto
            }
            : ProviderMode.Auto;

        return new Options
        {
            InputPath = Path.GetFullPath(input),
            BackgroundPath = Path.GetFullPath(background),
            OutputPath = Path.GetFullPath(output),
            ModelPath = Path.GetFullPath(model),
            DownsampleRatio = ratio,
            Provider = provider
        };
    }

    private static string Require(Dictionary<string, string> map, string key)
    {
        if (!map.TryGetValue(key, out var value) || string.IsNullOrWhiteSpace(value))
        {
            throw new ArgumentException($"{key} is required.\n\n{HelpText}");
        }

        return value;
    }

    private static string FindDefaultModel(string root)
    {
        var candidates = new[]
        {
            Path.Combine(root, "models", "rvm_mobilenetv3_fp32.onnx"),
            Path.GetFullPath(Path.Combine(root, "..", "native_matting_client", "models", "rvm_mobilenetv3_fp32.onnx")),
            Path.GetFullPath(Path.Combine(root, "..", "depth_matting_client", "models", "rvm_mobilenetv3_fp32.onnx"))
        };

        var found = candidates.FirstOrDefault(File.Exists);
        if (found == null)
        {
            throw new FileNotFoundException("RVM ONNX model not found. Use --model <path>.");
        }

        return found;
    }
}

internal enum ProviderMode
{
    Auto,
    DirectML,
    Cpu
}

public sealed class RvmOnnxMatting : IDisposable
{
    private readonly InferenceSession _session;
    private DenseTensor<float>[] _rec = CreateInitialRec();
    private readonly string[] _inputNames;
    public string ProviderName { get; }

    public RvmOnnxMatting(string modelPath, bool useGpu)
    {
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            EnableMemoryPattern = false
        };

        if (useGpu)
        {
            options.AppendExecutionProvider_DML(0);
            ProviderName = "DirectML";
        }
        else
        {
            ProviderName = "CPU";
        }

        _session = new InferenceSession(modelPath, options);
        _inputNames = _session.InputMetadata.Keys.ToArray();
    }

    public Mat Matte(Mat bgr, float downsampleRatio, Mat bg)
    {
        var srcTensor = ToTensor(bgr);
        var downsample = new DenseTensor<float>(new[] { Math.Clamp(downsampleRatio, 0.05f, 1f) }, [1]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName(0, "src"), srcTensor),
            NamedOnnxValue.CreateFromTensor(InputName(1, "r1i"), _rec[0]),
            NamedOnnxValue.CreateFromTensor(InputName(2, "r2i"), _rec[1]),
            NamedOnnxValue.CreateFromTensor(InputName(3, "r3i"), _rec[2]),
            NamedOnnxValue.CreateFromTensor(InputName(4, "r4i"), _rec[3]),
            NamedOnnxValue.CreateFromTensor(InputName(5, "downsample_ratio"), downsample)
        };

        using var results = _session.Run(inputs);
        var resultList = results.ToList();
        var fgr = resultList[0].AsTensor<float>();
        var pha = resultList[1].AsTensor<float>();
        _rec = resultList.Skip(2).Take(4).Select(v => CloneTensor(v.AsTensor<float>())).ToArray();
        return Composite(fgr, pha, bg);
    }

    private string InputName(int index, string fallback) => _inputNames.Length > index ? _inputNames[index] : fallback;

    private static DenseTensor<float>[] CreateInitialRec() =>
    [
        new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1]),
        new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1]),
        new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1]),
        new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1])
    ];

    private static DenseTensor<float> CloneTensor(Tensor<float> tensor)
    {
        var clone = new DenseTensor<float>(tensor.Dimensions.ToArray());
        tensor.ToArray().CopyTo(clone.Buffer.Span);
        return clone;
    }

    private static DenseTensor<float> ToTensor(Mat bgr)
    {
        var height = bgr.Height;
        var width = bgr.Width;
        var tensor = new DenseTensor<float>([1, 3, height, width]);
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var p = bgr.At<Vec3b>(y, x);
                tensor[0, 0, y, x] = p.Item2 / 255f;
                tensor[0, 1, y, x] = p.Item1 / 255f;
                tensor[0, 2, y, x] = p.Item0 / 255f;
            }
        }

        return tensor;
    }

    private static Mat Composite(Tensor<float> fgr, Tensor<float> pha, Mat bg)
    {
        var height = bg.Height;
        var width = bg.Width;
        var output = new Mat(height, width, MatType.CV_8UC3);
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var alpha = Math.Clamp(pha[0, 0, y, x], 0f, 1f);
                var bgPixel = bg.At<Vec3b>(y, x);
                var fr = Math.Clamp(fgr[0, 0, y, x], 0f, 1f) * 255f;
                var fg = Math.Clamp(fgr[0, 1, y, x], 0f, 1f) * 255f;
                var fb = Math.Clamp(fgr[0, 2, y, x], 0f, 1f) * 255f;
                output.Set(y, x, new Vec3b(
                    (byte)Math.Clamp(fb * alpha + bgPixel.Item0 * (1f - alpha), 0f, 255f),
                    (byte)Math.Clamp(fg * alpha + bgPixel.Item1 * (1f - alpha), 0f, 255f),
                    (byte)Math.Clamp(fr * alpha + bgPixel.Item2 * (1f - alpha), 0f, 255f)));
            }
        }

        return output;
    }

    public void Dispose() => _session.Dispose();
}

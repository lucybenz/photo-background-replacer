using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using CvSize = OpenCvSharp.Size;
using WinImage = System.Drawing.Image;

namespace PhotoBackgroundReplacerUI;

public partial class Form1 : Form
{
    private readonly string _rootDir;
    private readonly string _outputDir;

    private readonly Button _pickPersonButton = new() { Text = "Select Person Photo", Width = 180, Height = 48 };
    private readonly Button _pickBackgroundButton = new() { Text = "Select Background", Width = 180, Height = 48 };
    private readonly Button _pickModelButton = new() { Text = "Select ONNX Model", Width = 180, Height = 48 };
    private readonly Button _runButton = new() { Text = "Replace Background", Width = 190, Height = 48 };
    private readonly Button _openOutputButton = new() { Text = "Open Output Folder", Width = 185, Height = 48 };
    private readonly ComboBox _providerBox = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 130, Height = 48 };
    private readonly NumericUpDown _downsampleInput = new() { Minimum = 0.05M, Maximum = 1.0M, DecimalPlaces = 3, Increment = 0.025M, Value = 0.125M, Width = 110, Height = 48 };
    private readonly Label _status = new() { AutoSize = false, Dock = DockStyle.Bottom, Height = 38, TextAlign = ContentAlignment.MiddleLeft };
    private readonly PictureBox _personPreview = new() { Dock = DockStyle.Fill, BackColor = Color.FromArgb(22, 22, 22), SizeMode = PictureBoxSizeMode.Zoom };
    private readonly PictureBox _backgroundPreview = new() { Dock = DockStyle.Fill, BackColor = Color.FromArgb(22, 22, 22), SizeMode = PictureBoxSizeMode.Zoom };
    private readonly PictureBox _resultPreview = new() { Dock = DockStyle.Fill, BackColor = Color.FromArgb(22, 22, 22), SizeMode = PictureBoxSizeMode.Zoom };

    private string? _personPath;
    private string? _backgroundPath;
    private string? _modelPath;
    private string? _lastOutputPath;

    public Form1()
    {
        InitializeComponent();
        _rootDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
        _modelPath = FindDefaultModel(_rootDir);
        _outputDir = Path.Combine(_rootDir, "ui_outputs");
        Directory.CreateDirectory(_outputDir);

        Text = "Photo Background Replacer - RVM DirectML";
        Width = 1500;
        Height = 900;
        MinimumSize = new System.Drawing.Size(1180, 720);

        BuildUi();
        WireEvents();
        SetStatus(_modelPath == null
            ? "Ready. RVM ONNX model not found. Click Select ONNX Model before processing."
            : $"Ready. Model: {Path.GetFileName(_modelPath)}");
    }

    private void BuildUi()
    {
        Font = new Font("Segoe UI", 11F, FontStyle.Regular, GraphicsUnit.Point);
        _providerBox.Items.AddRange(["Auto", "DirectML", "CPU"]);
        _providerBox.SelectedIndex = 0;

        var top = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 118,
            Padding = new Padding(12),
            WrapContents = true,
            BackColor = Color.FromArgb(245, 245, 245)
        };

        top.Controls.AddRange([
            _pickPersonButton,
            _pickBackgroundButton,
            _pickModelButton,
            LabelOf("Provider"), _providerBox,
            LabelOf("Quality S"), _downsampleInput,
            _runButton,
            _openOutputButton
        ]);

        var grid = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 2 };
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.33F));
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.33F));
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.34F));
        grid.RowStyles.Add(new RowStyle(SizeType.Absolute, 34));
        grid.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        grid.Controls.Add(Header("Person Photo"), 0, 0);
        grid.Controls.Add(Header("Background"), 1, 0);
        grid.Controls.Add(Header("Result"), 2, 0);
        grid.Controls.Add(_personPreview, 0, 1);
        grid.Controls.Add(_backgroundPreview, 1, 1);
        grid.Controls.Add(_resultPreview, 2, 1);

        Controls.Add(grid);
        Controls.Add(_status);
        Controls.Add(top);
    }

    private static Label LabelOf(string text) => new()
    {
        Text = text,
        AutoSize = false,
        Width = 90,
        Height = 48,
        TextAlign = ContentAlignment.MiddleCenter,
        Margin = new Padding(10, 3, 2, 8)
    };

    private static Label Header(string text) => new()
    {
        Text = text,
        Dock = DockStyle.Fill,
        TextAlign = ContentAlignment.MiddleCenter,
        BackColor = Color.FromArgb(232, 232, 232)
    };

    private void WireEvents()
    {
        _pickPersonButton.Click += (_, _) => PickImage(isPerson: true);
        _pickBackgroundButton.Click += (_, _) => PickImage(isPerson: false);
        _pickModelButton.Click += (_, _) => PickModel();
        _runButton.Click += async (_, _) => await RunReplacementAsync();
        _openOutputButton.Click += (_, _) => OpenPath(_lastOutputPath ?? _outputDir);
    }

    private void PickImage(bool isPerson)
    {
        using var dialog = new OpenFileDialog
        {
            Filter = "Image|*.jpg;*.jpeg;*.png;*.bmp;*.webp|All files|*.*"
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        if (isPerson)
        {
            _personPath = dialog.FileName;
            LoadPreview(_personPreview, dialog.FileName);
            SetStatus($"Person photo selected: {dialog.FileName}");
        }
        else
        {
            _backgroundPath = dialog.FileName;
            LoadPreview(_backgroundPreview, dialog.FileName);
            SetStatus($"Background selected: {dialog.FileName}");
        }
    }

    private void PickModel()
    {
        using var dialog = new OpenFileDialog
        {
            Filter = "ONNX model|*.onnx|All files|*.*"
        };

        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        _modelPath = dialog.FileName;
        SetStatus($"Model selected: {dialog.FileName}");
    }

    private static void LoadPreview(PictureBox target, string path)
    {
        using var src = Cv2.ImRead(path, ImreadModes.Color);
        if (src.Empty())
        {
            return;
        }

        using var preview = ResizeLongest(src, 1400);
        var bmp = BitmapConverter.ToBitmap(preview);
        var old = target.Image;
        target.Image = bmp;
        old?.Dispose();
    }

    private async Task RunReplacementAsync()
    {
        if (string.IsNullOrWhiteSpace(_personPath) || string.IsNullOrWhiteSpace(_backgroundPath))
        {
            SetStatus("Select both a person photo and a background image first.");
            return;
        }

        if (string.IsNullOrWhiteSpace(_modelPath) || !File.Exists(_modelPath))
        {
            SetStatus("RVM ONNX model not found. Click Select ONNX Model and choose rvm_mobilenetv3_fp32.onnx.");
            return;
        }

        _runButton.Enabled = false;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        SetStatus("Replacing background. Full-resolution processing may take a while...");
        try
        {
            var personPath = _personPath;
            var backgroundPath = _backgroundPath;
            var outputPath = Path.Combine(_outputDir, $"photo_bg_result_{DateTime.Now:yyyyMMdd_HHmmss}.png");
            var provider = _providerBox.SelectedIndex;
            var downsample = (float)_downsampleInput.Value;
            var modelPath = _modelPath;

            var result = await Task.Run(() => ProcessImage(personPath, backgroundPath, outputPath, modelPath, provider, downsample));
            clock.Stop();
            _lastOutputPath = result.OutputPath;
            LoadPreview(_resultPreview, result.OutputPath);
            SetStatus($"Saved result: {result.OutputPath} | Provider {result.ProviderName} | {clock.Elapsed.TotalSeconds:F1}s");
            OpenPath(result.OutputPath);
        }
        catch (Exception ex)
        {
            SetStatus($"Failed: {ex.Message}");
            MessageBox.Show(this, ex.ToString(), "Processing failed", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
        finally
        {
            _runButton.Enabled = true;
        }
    }

    private static ProcessingResult ProcessImage(string personPath, string backgroundPath, string outputPath, string modelPath, int provider, float downsample)
    {
        using var person = Cv2.ImRead(personPath, ImreadModes.Color);
        using var background = Cv2.ImRead(backgroundPath, ImreadModes.Color);
        if (person.Empty())
        {
            throw new FileNotFoundException("Person photo cannot be read.", personPath);
        }

        if (background.Empty())
        {
            throw new FileNotFoundException("Background image cannot be read.", backgroundPath);
        }

        using var bg = CoverResize(background, person.Width, person.Height);
        using var result = RunMattingWithFallback(person, bg, modelPath, provider, downsample, out var providerName);
        Cv2.ImWrite(outputPath, result);
        return new ProcessingResult(outputPath, providerName);
    }

    private static Mat RunMattingWithFallback(Mat person, Mat bg, string modelPath, int provider, float downsample, out string providerName)
    {
        var providers = provider switch
        {
            1 => new[] { true },
            2 => new[] { false },
            _ => new[] { true, false }
        };
        var errors = new List<string>();
        foreach (var useGpu in providers)
        {
            try
            {
                using var model = new RvmOnnxMatting(modelPath, useGpu);
                providerName = model.ProviderName;
                return model.Matte(person, downsample, bg);
            }
            catch (Exception ex)
            {
                errors.Add($"{(useGpu ? "DirectML" : "CPU")}: {ex.Message}");
            }
        }

        providerName = "Unavailable";
        throw new InvalidOperationException("Matting failed. " + string.Join(" | ", errors));
    }

    private static Mat CoverResize(Mat src, int targetWidth, int targetHeight)
    {
        var scale = Math.Max(targetWidth / (double)src.Width, targetHeight / (double)src.Height);
        using var resized = new Mat();
        Cv2.Resize(src, resized, new CvSize((int)Math.Ceiling(src.Width * scale), (int)Math.Ceiling(src.Height * scale)), 0, 0, InterpolationFlags.Linear);
        var x = Math.Max(0, (resized.Width - targetWidth) / 2);
        var y = Math.Max(0, (resized.Height - targetHeight) / 2);
        return new Mat(resized, new Rect(x, y, targetWidth, targetHeight)).Clone();
    }

    private static Mat ResizeLongest(Mat src, int maxSide)
    {
        var longest = Math.Max(src.Width, src.Height);
        if (longest <= maxSide)
        {
            return src.Clone();
        }

        var scale = maxSide / (double)longest;
        var dst = new Mat();
        Cv2.Resize(src, dst, new CvSize((int)Math.Round(src.Width * scale), (int)Math.Round(src.Height * scale)), 0, 0, InterpolationFlags.Area);
        return dst;
    }

    private static string? FindDefaultModel(string root)
    {
        var candidates = new[]
        {
            Path.Combine(root, "models", "rvm_mobilenetv3_fp32.onnx"),
            Path.GetFullPath(Path.Combine(root, "..", "native_matting_client", "models", "rvm_mobilenetv3_fp32.onnx")),
            Path.GetFullPath(Path.Combine(root, "..", "depth_matting_client", "models", "rvm_mobilenetv3_fp32.onnx"))
        };

        return candidates.FirstOrDefault(File.Exists);
    }

    private static void OpenPath(string path)
    {
        try
        {
            var args = Directory.Exists(path) ? $"\"{path}\"" : $"/select,\"{path}\"";
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = args,
                UseShellExecute = true
            });
        }
        catch
        {
            // Convenience only.
        }
    }

    private void SetStatus(string text) => _status.Text = "  " + text;
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

public sealed record ProcessingResult(string OutputPath, string ProviderName);

#include <acl/acl.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kDeviceId = 0;
constexpr int kModelW = 640;
constexpr int kModelH = 640;
constexpr int kChannels = 3;
constexpr float kConfThresh = 0.25f;

// 视频源
//const std::string kVideoSource = "/home/HwHiAiUser/YOLO26/transportation.mp4";
const std::string kVideoSource = "0";

// AIPP 版模型路径
const char* kModelPath = "/home/HwHiAiUser/YOLO26/yolo26nfp16_aipp_opt3.om";
const char* kSaveVideoPath = "/home/HwHiAiUser/YOLO26/result_video.mp4";
// COCO 80 classes
const std::vector<std::string> kCocoClasses = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
};

void CheckAcl(const std::string& name, aclError ret) {
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error(name + " failed, ret=" + std::to_string(ret));
    }
}

struct Detection {
    float x1;
    float y1;
    float x2;
    float y2;
    float conf;
    int cls;
};

cv::Scalar GetClassColor(int cls) {
    static const std::vector<cv::Scalar> palette = {
        {255,  56,  56}, {255, 157, 151}, {255, 112,  31}, {255, 178,  29},
        {207, 210,  49}, { 72, 249,  10}, {146, 204,  23}, { 61, 219, 134},
        { 26, 147,  52}, {  0, 212, 187}, { 44, 153, 168}, {  0, 194, 255},
        { 52,  69, 147}, {100, 115, 255}, {  0,  24, 236}, {132,  56, 255},
        { 82,   0, 133}, {203,  56, 255}, {255, 149, 200}, {255,  55, 199},
        { 90, 180, 255}, {180,  90, 255}, {255, 200,  90}, {120, 255, 120},
        {255, 120, 120}, {120, 120, 255}, {200, 255, 120}, {255, 120, 220},
        {120, 255, 220}, {220, 120, 255}
    };
    if (cls < 0) return cv::Scalar(0, 255, 0);
    return palette[cls % palette.size()];
}

class Preprocessor {
public:
    Preprocessor() : input_(static_cast<size_t>(kModelW) * kModelH * kChannels) {
        resized_.create(kModelH, kModelW, CV_8UC3);
        rgb_.create(kModelH, kModelW, CV_8UC3);
    }

    const std::vector<uint8_t>& Run(const cv::Mat& bgr) {
        cv::resize(bgr, resized_, cv::Size(kModelW, kModelH), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(resized_, rgb_, cv::COLOR_BGR2RGB);

        const size_t bytes = static_cast<size_t>(rgb_.rows) * rgb_.cols * rgb_.channels();
        std::memcpy(input_.data(), rgb_.data, bytes);
        return input_;
    }

private:
    cv::Mat resized_;
    cv::Mat rgb_;
    std::vector<uint8_t> input_;
};

cv::Rect ScaleBoxToOriginal(float x1, float y1, float x2, float y2, int origW, int origH) {
    auto clampf = [](float v, float lo, float hi) {
        return std::max(lo, std::min(v, hi));
    };

    x1 = clampf(x1, 0.0f, static_cast<float>(kModelW - 1));
    y1 = clampf(y1, 0.0f, static_cast<float>(kModelH - 1));
    x2 = clampf(x2, 0.0f, static_cast<float>(kModelW - 1));
    y2 = clampf(y2, 0.0f, static_cast<float>(kModelH - 1));

    float sx1 = x1 / static_cast<float>(kModelW) * origW;
    float sy1 = y1 / static_cast<float>(kModelH) * origH;
    float sx2 = x2 / static_cast<float>(kModelW) * origW;
    float sy2 = y2 / static_cast<float>(kModelH) * origH;

    int ix1 = std::max(0, std::min(static_cast<int>(std::round(sx1)), origW - 1));
    int iy1 = std::max(0, std::min(static_cast<int>(std::round(sy1)), origH - 1));
    int ix2 = std::max(0, std::min(static_cast<int>(std::round(sx2)), origW - 1));
    int iy2 = std::max(0, std::min(static_cast<int>(std::round(sy2)), origH - 1));

    int w = std::max(1, ix2 - ix1);
    int h = std::max(1, iy2 - iy1);
    return cv::Rect(ix1, iy1, w, h);
}

void DrawDetections(cv::Mat& image, const std::vector<Detection>& dets) {
    for (const auto& d : dets) {
        cv::Rect box = ScaleBoxToOriginal(d.x1, d.y1, d.x2, d.y2, image.cols, image.rows);
        cv::Scalar color = GetClassColor(d.cls);

        cv::rectangle(image, box, color, 2);

        std::string label = std::to_string(d.cls);
        if (d.cls >= 0 && d.cls < static_cast<int>(kCocoClasses.size())) {
            label = kCocoClasses[d.cls];
        }
        label += " " + cv::format("%.2f", d.conf);

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

        int tx = box.x;
        int ty = std::max(0, box.y - textSize.height - baseline - 4);
        int tw = textSize.width + 4;
        int th = textSize.height + baseline + 4;

        cv::rectangle(image, cv::Rect(tx, ty, tw, th), color, -1);
        cv::putText(image, label, cv::Point(tx + 2, ty + th - baseline - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2, cv::LINE_AA);
    }
}

bool IsCameraSource(const std::string& s) {
    if (s.empty()) return false;
    return std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); });
}

class FpsSmoother {
public:
    explicit FpsSmoother(size_t maxSize = 30) : maxSize_(maxSize) {}

    void Add(double value) {
        values_.push_back(value);
        if (values_.size() > maxSize_) values_.pop_front();
    }

    double Mean() const {
        if (values_.empty()) return 0.0;
        double sum = std::accumulate(values_.begin(), values_.end(), 0.0);
        return sum / static_cast<double>(values_.size());
    }

private:
    size_t maxSize_;
    std::deque<double> values_;
};

}  // namespace

int main() {
    aclrtContext context = nullptr;
    aclrtStream stream = nullptr;
    uint32_t modelId = 0;
    aclmdlDesc* modelDesc = nullptr;
    aclmdlDataset* inputDataset = nullptr;
    aclmdlDataset* outputDataset = nullptr;
    aclDataBuffer* inputBuffer = nullptr;
    aclDataBuffer* outputBuffer = nullptr;
    void* inputDev = nullptr;
    void* outputDev = nullptr;
    void* outputHost = nullptr;

    cv::VideoCapture cap;
    cv::VideoWriter writer;

    try {
        CheckAcl("aclInit", aclInit(nullptr));
        CheckAcl("aclrtSetDevice", aclrtSetDevice(kDeviceId));
        CheckAcl("aclrtCreateContext", aclrtCreateContext(&context, kDeviceId));
        CheckAcl("aclrtCreateStream", aclrtCreateStream(&stream));

        CheckAcl("aclmdlLoadFromFile", aclmdlLoadFromFile(kModelPath, &modelId));

        modelDesc = aclmdlCreateDesc();
        if (!modelDesc) throw std::runtime_error("aclmdlCreateDesc returned nullptr");
        CheckAcl("aclmdlGetDesc", aclmdlGetDesc(modelDesc, modelId));

        size_t numInputs = aclmdlGetNumInputs(modelDesc);
        size_t numOutputs = aclmdlGetNumOutputs(modelDesc);

        std::cout << "num_inputs = " << numInputs << std::endl;
        std::cout << "num_outputs = " << numOutputs << std::endl;

        for (size_t i = 0; i < numInputs; ++i) {
            size_t size = aclmdlGetInputSizeByIndex(modelDesc, i);
            aclmdlIODims dims{};
            aclmdlGetInputDims(modelDesc, i, &dims);
            aclDataType dtype = aclmdlGetInputDataType(modelDesc, i);

            std::cout << "[INPUT " << i << "] size=" << size << ", dims=[";
            for (size_t j = 0; j < dims.dimCount; ++j) {
                std::cout << dims.dims[j] << (j + 1 < dims.dimCount ? "," : "");
            }
            std::cout << "], dtype=" << static_cast<int>(dtype) << std::endl;
        }

        for (size_t i = 0; i < numOutputs; ++i) {
            size_t size = aclmdlGetOutputSizeByIndex(modelDesc, i);
            aclmdlIODims dims{};
            aclmdlGetOutputDims(modelDesc, i, &dims);
            aclDataType dtype = aclmdlGetOutputDataType(modelDesc, i);

            std::cout << "[OUTPUT " << i << "] size=" << size << ", dims=[";
            for (size_t j = 0; j < dims.dimCount; ++j) {
                std::cout << dims.dims[j] << (j + 1 < dims.dimCount ? "," : "");
            }
            std::cout << "], dtype=" << static_cast<int>(dtype) << std::endl;
        }

        const size_t inputSize = aclmdlGetInputSizeByIndex(modelDesc, 0);
        const size_t outputSize = aclmdlGetOutputSizeByIndex(modelDesc, 0);

        CheckAcl("aclrtMalloc(input)", aclrtMalloc(&inputDev, inputSize, ACL_MEM_MALLOC_NORMAL_ONLY));
        CheckAcl("aclrtMalloc(output)", aclrtMalloc(&outputDev, outputSize, ACL_MEM_MALLOC_NORMAL_ONLY));
        CheckAcl("aclrtMallocHost(outputHost)", aclrtMallocHost(&outputHost, outputSize));

        inputDataset = aclmdlCreateDataset();
        outputDataset = aclmdlCreateDataset();
        if (!inputDataset || !outputDataset) throw std::runtime_error("aclmdlCreateDataset failed");

        inputBuffer = aclCreateDataBuffer(inputDev, inputSize);
        outputBuffer = aclCreateDataBuffer(outputDev, outputSize);
        if (!inputBuffer || !outputBuffer) throw std::runtime_error("aclCreateDataBuffer failed");

        CheckAcl("aclmdlAddDatasetBuffer(input)", aclmdlAddDatasetBuffer(inputDataset, inputBuffer));
        CheckAcl("aclmdlAddDatasetBuffer(output)", aclmdlAddDatasetBuffer(outputDataset, outputBuffer));

        if (IsCameraSource(kVideoSource)) {
            int camId = std::stoi(kVideoSource);
            cap.open(camId, cv::CAP_V4L2);
            cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, 640);
            
        } else {
            cap.open(kVideoSource);
        }

        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video source: " + kVideoSource);
        }

        int frameW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int frameH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        double inputFps = cap.get(cv::CAP_PROP_FPS);
        if (inputFps <= 0.0 || std::isnan(inputFps)) inputFps = 25.0;

        std::cout << "Video source opened. frameW=" << frameW
                  << ", frameH=" << frameH
                  << ", inputFps=" << inputFps << std::endl;

        bool enableSave = false;
        if (enableSave) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(kSaveVideoPath, fourcc, inputFps, cv::Size(frameW, frameH));
            if (!writer.isOpened()) {
                std::cerr << "[WARN] Failed to open writer: " << kSaveVideoPath << std::endl;
            }
        }

        Preprocessor preprocessor;
        FpsSmoother modelFpsSmooth(30);
        FpsSmoother e2eFpsSmooth(30);

        cv::Mat frame;
        size_t frameIdx = 0;

        while (true) {
            if (!cap.read(frame) || frame.empty()) {
                std::cout << "End of stream or failed to read frame." << std::endl;
                break;
            }

            auto t0E2E = std::chrono::high_resolution_clock::now();

            auto t0Pre = std::chrono::high_resolution_clock::now();
            const std::vector<uint8_t>& input = preprocessor.Run(frame);
            auto t1Pre = std::chrono::high_resolution_clock::now();

            if (input.size() != inputSize) {
                throw std::runtime_error(
                    "Input size mismatch, host=" + std::to_string(input.size()) +
                    ", model=" + std::to_string(inputSize));
            }

            CheckAcl("aclrtMemcpy H2D",
                     aclrtMemcpy(inputDev, inputSize, input.data(), inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

            auto t0Exec = std::chrono::high_resolution_clock::now();
            CheckAcl("aclmdlExecute", aclmdlExecute(modelId, inputDataset, outputDataset));
            auto t1Exec = std::chrono::high_resolution_clock::now();

            auto t0Post = std::chrono::high_resolution_clock::now();

            CheckAcl("aclrtMemcpy D2H",
                     aclrtMemcpy(outputHost, outputSize, outputDev, outputSize, ACL_MEMCPY_DEVICE_TO_HOST));

            const float* out = reinterpret_cast<const float*>(outputHost);
            const size_t detCount = 300;
            const size_t detStride = 6;

            std::vector<Detection> detections;
            detections.reserve(detCount);

            for (size_t i = 0; i < detCount; ++i) {
                const float* p = out + i * detStride;
                float conf = p[4];
                if (conf <= kConfThresh) continue;

                int cls = static_cast<int>(std::round(p[5]));
                if (cls < 0 || cls >= static_cast<int>(kCocoClasses.size())) continue;

                Detection d;
                d.x1 = p[0];
                d.y1 = p[1];
                d.x2 = p[2];
                d.y2 = p[3];
                d.conf = conf;
                d.cls = cls;

                if (d.x2 <= d.x1 || d.y2 <= d.y1) continue;
                detections.push_back(d);
            }

            //cv::Mat vis = frame.clone();
            DrawDetections(frame, detections);

            auto t1Post = std::chrono::high_resolution_clock::now();
            auto t1E2E = std::chrono::high_resolution_clock::now();

            double preprocessMs = std::chrono::duration<double, std::milli>(t1Pre - t0Pre).count();
            double executeMs = std::chrono::duration<double, std::milli>(t1Exec - t0Exec).count();
            double postprocessMs = std::chrono::duration<double, std::milli>(t1Post - t0Post).count();
            double e2eMs = std::chrono::duration<double, std::milli>(t1E2E - t0E2E).count();

            double modelFps = 1000.0 / executeMs;
            double e2eFps = 1000.0 / e2eMs;

            modelFpsSmooth.Add(modelFps);
            e2eFpsSmooth.Add(e2eFps);

            std::string line1 = cv::format("Model FPS: %.2f  E2E FPS: %.2f",
                                           modelFpsSmooth.Mean(), e2eFpsSmooth.Mean());
            std::string line2 = cv::format("Pre: %.2f ms  Exec: %.2f ms  Post: %.2f ms",
                                           preprocessMs, executeMs, postprocessMs);
            std::string line3 = cv::format("Detections: %zu  Frame: %zu",
                                           detections.size(), frameIdx);

            cv::putText(frame, line1, cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::putText(frame, line2, cv::Point(20, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            cv::putText(frame, line3, cv::Point(20, 90),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

            if (enableSave && writer.isOpened()) writer.write(frame);

            cv::imshow("YOLO26 Atlas Detection Demo", frame);
            int key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                std::cout << "Exit requested." << std::endl;
                break;
            }

            if (frameIdx % 60 == 0) {
                std::cout << "[frame " << frameIdx << "] "
                          << "pre=" << preprocessMs << " ms, "
                          << "exec=" << executeMs << " ms, "
                          << "post=" << postprocessMs << " ms, "
                          << "e2e=" << e2eMs << " ms, "
                          << "det=" << detections.size() << std::endl;
            }

            ++frameIdx;
        }

        cap.release();
        if (writer.isOpened()) writer.release();
        cv::destroyAllWindows();

        if (outputBuffer) { aclDestroyDataBuffer(outputBuffer); outputBuffer = nullptr; }
        if (inputBuffer) { aclDestroyDataBuffer(inputBuffer); inputBuffer = nullptr; }
        if (outputDataset) { aclmdlDestroyDataset(outputDataset); outputDataset = nullptr; }
        if (inputDataset) { aclmdlDestroyDataset(inputDataset); inputDataset = nullptr; }
        if (outputHost) { aclrtFreeHost(outputHost); outputHost = nullptr; }
        if (outputDev) { aclrtFree(outputDev); outputDev = nullptr; }
        if (inputDev) { aclrtFree(inputDev); inputDev = nullptr; }
        if (modelDesc) { aclmdlDestroyDesc(modelDesc); modelDesc = nullptr; }

        CheckAcl("aclmdlUnload", aclmdlUnload(modelId));
        modelId = 0;
        CheckAcl("aclrtDestroyStream", aclrtDestroyStream(stream));
        stream = nullptr;
        CheckAcl("aclrtDestroyContext", aclrtDestroyContext(context));
        context = nullptr;
        CheckAcl("aclrtResetDevice", aclrtResetDevice(kDeviceId));
        CheckAcl("aclFinalize", aclFinalize());

        std::cout << "Done." << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;

        if (cap.isOpened()) cap.release();
        if (writer.isOpened()) writer.release();
        cv::destroyAllWindows();

        if (outputBuffer) aclDestroyDataBuffer(outputBuffer);
        if (inputBuffer) aclDestroyDataBuffer(inputBuffer);
        if (outputDataset) aclmdlDestroyDataset(outputDataset);
        if (inputDataset) aclmdlDestroyDataset(inputDataset);
        if (outputHost) aclrtFreeHost(outputHost);
        if (outputDev) aclrtFree(outputDev);
        if (inputDev) aclrtFree(inputDev);
        if (modelDesc) aclmdlDestroyDesc(modelDesc);
        if (modelId != 0) aclmdlUnload(modelId);
        if (stream) aclrtDestroyStream(stream);
        if (context) aclrtDestroyContext(context);
        aclrtResetDevice(kDeviceId);
        aclFinalize();
        return 1;
    }
}

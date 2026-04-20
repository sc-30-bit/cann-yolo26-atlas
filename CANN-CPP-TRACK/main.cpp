#include <acl/acl.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

constexpr int kDeviceId = 0;
constexpr int kModelW = 640;
constexpr int kModelH = 640;
constexpr int kChannels = 3;

constexpr float kConfThresh = 0.45f;
constexpr float kTrackHighThresh = 0.45f;
constexpr float kTrackLowThresh  = 0.10f;
constexpr float kMatchIouThreshHigh = 0.30f;
constexpr float kMatchIouThreshLow  = 0.20f;
constexpr int   kTrackBuffer = 20;

// const std::string kVideoSource = "/home/HwHiAiUser/YOLO26/transportation.mp4";
const char* kModelPath = "/home/HwHiAiUser/YOLO26/yolo26nfp16_aipp_opt3.om";
const std::string kVideoSource = "0";
const char* kSaveVideoPath = "/home/HwHiAiUser/YOLO26/result_video.mp4";

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
    int track_id = -1;
};

struct Track {
    int id = -1;
    int cls = -1;
    cv::Rect2f box;
    float conf = 0.0f;
    int age = 0;
    int time_since_update = 0;
    bool matched_in_frame = false;
};

cv::Scalar GetTrackColor(int id) {
    if (id < 0) return cv::Scalar(0, 255, 0);
    int hue = (id * 37) % 180;
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 220, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    cv::Vec3b c = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(c[0], c[1], c[2]);
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

cv::Rect2f DetToRectF(const Detection& d) {
    float w = std::max(1.0f, d.x2 - d.x1);
    float h = std::max(1.0f, d.y2 - d.y1);
    return cv::Rect2f(d.x1, d.y1, w, h);
}

float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float xx1 = std::max(a.x, b.x);
    float yy1 = std::max(a.y, b.y);
    float xx2 = std::min(a.x + a.width,  b.x + b.width);
    float yy2 = std::min(a.y + a.height, b.y + b.height);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;
    float uni = a.width * a.height + b.width * b.height - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

class ByteTrackerLite {
public:
    ByteTrackerLite(float highThresh, float lowThresh, float matchIouHigh, float matchIouLow, int trackBuffer)
        : highThresh_(highThresh),
          lowThresh_(lowThresh),
          matchIouHigh_(matchIouHigh),
          matchIouLow_(matchIouLow),
          trackBuffer_(trackBuffer) {}

    void Update(std::vector<Detection>& detections) {
        for (auto& tr : tracks_) {
            tr.matched_in_frame = false;
            tr.time_since_update += 1;
            tr.age += 1;
        }

        std::vector<int> highDetIdx;
        std::vector<int> lowDetIdx;
        highDetIdx.reserve(detections.size());
        lowDetIdx.reserve(detections.size());

        for (size_t i = 0; i < detections.size(); ++i) {
            if (detections[i].conf >= highThresh_) highDetIdx.push_back(static_cast<int>(i));
            else if (detections[i].conf >= lowThresh_) lowDetIdx.push_back(static_cast<int>(i));
        }

        GreedyMatch(detections, highDetIdx, matchIouHigh_);
        GreedyMatchOnlyUnmatchedTracks(detections, lowDetIdx, matchIouLow_);

        for (int di : highDetIdx) {
            if (detections[di].track_id >= 0) continue;
            CreateTrack(detections[di]);
        }

        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                           [&](const Track& t) { return t.time_since_update > trackBuffer_; }),
            tracks_.end());
    }

private:
    void CreateTrack(Detection& det) {
        Track tr;
        tr.id = next_id_++;
        tr.cls = det.cls;
        tr.box = DetToRectF(det);
        tr.conf = det.conf;
        tr.age = 1;
        tr.time_since_update = 0;
        tr.matched_in_frame = true;
        tracks_.push_back(tr);
        det.track_id = tr.id;
    }

    void UpdateTrackFromDet(Track& tr, Detection& det) {
        tr.box = DetToRectF(det);
        tr.conf = det.conf;
        tr.cls = det.cls;
        tr.time_since_update = 0;
        tr.matched_in_frame = true;
        det.track_id = tr.id;
    }

    void GreedyMatch(std::vector<Detection>& detections, const std::vector<int>& detIndices, float iouThresh) {
        std::vector<bool> detUsed(detections.size(), false);
        for (size_t i = 0; i < detections.size(); ++i) {
            if (detections[i].track_id >= 0) detUsed[i] = true;
        }

        for (auto& tr : tracks_) {
            float bestIou = 0.0f;
            int bestDet = -1;

            for (int di : detIndices) {
                if (detUsed[di]) continue;
                if (detections[di].cls != tr.cls) continue;

                float iou = IoU(tr.box, DetToRectF(detections[di]));
                if (iou > bestIou) {
                    bestIou = iou;
                    bestDet = di;
                }
            }

            if (bestDet >= 0 && bestIou >= iouThresh) {
                UpdateTrackFromDet(tr, detections[bestDet]);
                detUsed[bestDet] = true;
            }
        }
    }

    void GreedyMatchOnlyUnmatchedTracks(std::vector<Detection>& detections,
                                        const std::vector<int>& detIndices,
                                        float iouThresh) {
        std::vector<bool> detUsed(detections.size(), false);
        for (size_t i = 0; i < detections.size(); ++i) {
            if (detections[i].track_id >= 0) detUsed[i] = true;
        }

        for (auto& tr : tracks_) {
            if (tr.matched_in_frame) continue;

            float bestIou = 0.0f;
            int bestDet = -1;

            for (int di : detIndices) {
                if (detUsed[di]) continue;
                if (detections[di].cls != tr.cls) continue;

                float iou = IoU(tr.box, DetToRectF(detections[di]));
                if (iou > bestIou) {
                    bestIou = iou;
                    bestDet = di;
                }
            }

            if (bestDet >= 0 && bestIou >= iouThresh) {
                UpdateTrackFromDet(tr, detections[bestDet]);
                detUsed[bestDet] = true;
            }
        }
    }

private:
    float highThresh_;
    float lowThresh_;
    float matchIouHigh_;
    float matchIouLow_;
    int trackBuffer_;
    int next_id_ = 0;
    std::vector<Track> tracks_;
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
        if (d.conf < kConfThresh) continue;

        cv::Rect box = ScaleBoxToOriginal(d.x1, d.y1, d.x2, d.y2, image.cols, image.rows);
        cv::Scalar color = GetTrackColor(d.track_id);

        cv::rectangle(image, box, color, 2);

        std::string label = std::to_string(d.cls);
        if (d.cls >= 0 && d.cls < static_cast<int>(kCocoClasses.size())) {
            label = kCocoClasses[d.cls];
        }
        if (d.track_id >= 0) {
            label += " ID:" + std::to_string(d.track_id);
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
        ByteTrackerLite tracker(
            kTrackHighThresh, kTrackLowThresh,
            kMatchIouThreshHigh, kMatchIouThreshLow,
            kTrackBuffer
        );
        FpsSmoother modelFpsSmooth(30);
        FpsSmoother e2eFpsSmooth(30);

        cv::Mat frame;
        size_t frameIdx = 0;
        cv::namedWindow("YOLO26 Atlas Tracking Demo", cv::WINDOW_NORMAL);

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
                if (conf < kTrackLowThresh) continue;

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

            tracker.Update(detections);

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

            size_t shownCount = 0;
            for (const auto& d : detections) {
                if (d.conf >= kConfThresh) ++shownCount;
            }

            std::string line1 = cv::format("Model FPS: %.2f  E2E FPS: %.2f",
                                           modelFpsSmooth.Mean(), e2eFpsSmooth.Mean());
            std::string line2 = cv::format("Pre: %.2f ms  Exec: %.2f ms  Post: %.2f ms",
                                           preprocessMs, executeMs, postprocessMs);
            std::string line3 = cv::format("Detections: %zu  Frame: %zu",
                                           shownCount, frameIdx);

            cv::putText(frame, line1, cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::putText(frame, line2, cv::Point(20, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
            cv::putText(frame, line3, cv::Point(20, 90),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

            if (enableSave && writer.isOpened()) writer.write(frame);

            cv::imshow("YOLO26 Atlas Tracking Demo", frame);
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
                          << "shown_det=" << shownCount << std::endl;
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

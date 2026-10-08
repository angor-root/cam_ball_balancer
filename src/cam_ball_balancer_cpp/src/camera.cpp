#include "cam_ball_balancer_cpp/camera.hpp"

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace cbb {

namespace {
std::string fourcc_str(double v) {
  const int f = static_cast<int>(v);
  return {char(f & 0xFF), char((f >> 8) & 0xFF), char((f >> 16) & 0xFF), char((f >> 24) & 0xFF)};
}
}  // namespace

Camera::Camera(const CameraConfig& cfg) {
  const bool is_index = !cfg.device.empty() &&
                        cfg.device.find_first_not_of("0123456789") == std::string::npos;
  if (is_index)
    cap_.open(std::stoi(cfg.device), cv::CAP_V4L2);
  else
    cap_.open(cfg.device, cv::CAP_V4L2);
  if (!cap_.isOpened()) throw std::runtime_error("No se pudo abrir la camara " + cfg.device);

  // Orden V4L2: formato antes que tamano/FPS.
  if (cfg.fourcc.size() == 4)
    cap_.set(cv::CAP_PROP_FOURCC,
             cv::VideoWriter::fourcc(cfg.fourcc[0], cfg.fourcc[1], cfg.fourcc[2], cfg.fourcc[3]));
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
  cap_.set(cv::CAP_PROP_FPS, cfg.fps);
  cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
  if (cfg.exposure > 0) {
    cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 1);  // V4L2 manual
    cap_.set(cv::CAP_PROP_EXPOSURE, cfg.exposure);
  } else {
    cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, 3);
  }
  if (cfg.wb_temperature > 0) {
    cap_.set(cv::CAP_PROP_AUTO_WB, 0);
    cap_.set(cv::CAP_PROP_WB_TEMPERATURE, cfg.wb_temperature);
  } else {
    cap_.set(cv::CAP_PROP_AUTO_WB, 1);
  }
}

Camera::~Camera() { stop(); }

std::string Camera::actual_settings() {
  char buf[256];
  std::snprintf(buf, sizeof buf, "%s %dx%d @%.1f fps, auto_exp=%.0f exp=%.0f, auto_wb=%.0f wb=%.0f",
                fourcc_str(cap_.get(cv::CAP_PROP_FOURCC)).c_str(),
                int(cap_.get(cv::CAP_PROP_FRAME_WIDTH)), int(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)),
                cap_.get(cv::CAP_PROP_FPS), cap_.get(cv::CAP_PROP_AUTO_EXPOSURE),
                cap_.get(cv::CAP_PROP_EXPOSURE), cap_.get(cv::CAP_PROP_AUTO_WB),
                cap_.get(cv::CAP_PROP_WB_TEMPERATURE));
  return buf;
}

void Camera::start() {
  running_ = true;
  thread_ = std::thread(&Camera::reader, this);
}

void Camera::stop() {
  if (!running_.exchange(false)) {
    if (cap_.isOpened()) cap_.release();
    return;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
  cap_.release();
}

void Camera::reader() {
  cv::Mat img;
  while (running_) {
    if (!cap_.read(img)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      continue;
    }
    Frame f;
    f.t_mono = now_s();
    f.t_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::system_clock::now().time_since_epoch()).count();
    f.seq = seq_++;
    rate.tick(f.t_mono);
    {
      std::lock_guard<std::mutex> lk(mtx_);
      f.image = img;  // img se reasigna en el siguiente read(); el consumidor copia via swap
      img = cv::Mat();
      latest_ = std::move(f);
    }
    cv_.notify_all();
  }
}

bool Camera::wait_next(long after_seq, Frame& out, double timeout_s) {
  std::unique_lock<std::mutex> lk(mtx_);
  const bool ok = cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), [&] {
    return !running_ || latest_.seq > after_seq;
  });
  if (!ok || !running_) return false;
  out = latest_;  // comparte el buffer (cv::Mat refcount); el lector ya usa uno nuevo
  return true;
}

}  // namespace cbb

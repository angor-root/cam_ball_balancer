// Captura V4L2 con hilo lector que conserva solo el cuadro mas reciente.
// Mismo diseno que camera.py (ver ahi el diagnostico de los ~11 FPS).
#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include <opencv2/videoio.hpp>

#include "cam_ball_balancer_cpp/vision_core.hpp"

namespace cbb {

struct CameraConfig {
  std::string device = "0";  // indice o /dev/videoN
  int width = 640, height = 480;
  double fps = 30.0;
  std::string fourcc = "MJPG";
  double exposure = -1;        // unidades de 100 us; <= 0 auto
  double wb_temperature = -1;  // K; <= 0 auto
};

struct Frame {
  cv::Mat image;
  long seq = -1;
  double t_mono = 0;        // steady_clock, s
  int64_t t_wall_ns = 0;    // system_clock, ns (stamp ROS)
};

class Camera {
 public:
  explicit Camera(const CameraConfig& cfg);
  ~Camera();
  std::string actual_settings();
  void start();
  void stop();
  // Espera un cuadro con seq > after_seq; false si timeout.
  bool wait_next(long after_seq, Frame& out, double timeout_s = 1.0);
  RateStats rate{120};  // tasa a la que la camara ENTREGA cuadros (ventana deslizante)
  long frames_read() const { return seq_; }

 private:
  void reader();
  cv::VideoCapture cap_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::mutex mtx_;
  std::condition_variable cv_;
  Frame latest_;
  std::atomic<long> seq_{0};
};

}  // namespace cbb

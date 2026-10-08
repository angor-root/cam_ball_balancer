// Mide la frecuencia real de la camara y del pipeline C++, sin ROS.
//
//   ros2 run cam_ball_balancer_cpp bench_vision --device 0 --sweep
//   ros2 run cam_ball_balancer_cpp bench_vision --device 0 --seconds 20
//        [--exposure 100] [--aruco-every 2] [--ids 0,1,3,2]
//        [--pos -0.035,-0.0415,0.035,-0.0415,0.035,0.0415,-0.035,0.0415]
//   ros2 run cam_ball_balancer_cpp bench_vision --image cuadro.png --iters 1000
//        (solo computo, sin camara: compara directo contra la version Python)
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/imgcodecs.hpp>

#include "cam_ball_balancer_cpp/camera.hpp"
#include "cam_ball_balancer_cpp/vision_core.hpp"

using namespace cbb;

namespace {

std::vector<double> parse_list(const std::string& s) {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::stod(tok));
  return v;
}

double pct(std::vector<double> a, double q) {
  if (a.empty()) return 0;
  std::sort(a.begin(), a.end());
  return a[std::min(a.size() - 1, size_t(q / 100.0 * (a.size() - 1) + 0.5))];
}

struct CamRun { double hz, jitter, max_ms; std::string got; };

CamRun capture_only(const CameraConfig& cc, double seconds) {
  Camera cam(cc);
  const std::string got = cam.actual_settings();
  cam.start();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  RateStats rs(1000000);
  Frame f;
  long last = -1;
  const double end = now_s() + seconds;
  while (now_s() < end) {
    if (!cam.wait_next(last, f)) continue;
    // cuenta tambien los que no alcanzamos a ver: usamos el reloj de captura
    last = f.seq;
    rs.tick(f.t_mono);
  }
  cam.stop();
  auto s = rs.summary();
  return {s.hz, s.jitter_ms, s.period_max_ms, got};
}

}  // namespace

int main(int argc, char** argv) {
  CameraConfig cc;
  double seconds = 10;
  bool sweep = false;
  std::string image_path;
  int iters = 1000;
  PipelineConfig pc;
  std::string intr_path;
  try {
    intr_path = ament_index_cpp::get_package_share_directory("cam_ball_balancer_cpp") +
                "/config/camera_intrinsics.yml";
  } catch (...) {}

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--device") cc.device = next();
    else if (a == "--width") cc.width = std::stoi(next());
    else if (a == "--height") cc.height = std::stoi(next());
    else if (a == "--fps") cc.fps = std::stod(next());
    else if (a == "--fourcc") cc.fourcc = next();
    else if (a == "--exposure") cc.exposure = std::stod(next());
    else if (a == "--wb") cc.wb_temperature = std::stod(next());
    else if (a == "--seconds") seconds = std::stod(next());
    else if (a == "--intrinsics") intr_path = next();
    else if (a == "--aruco-every") pc.aruco_every_n = std::stoi(next());
    else if (a == "--no-roi") pc.use_roi = false;
    else if (a == "--aruco-downscale") pc.pose.aruco_downscale = std::stoi(next());
    else if (a == "--thresh") {
      auto v = parse_list(next());
      pc.pose.thresh_win_min = int(v.at(0)); pc.pose.thresh_win_max = int(v.at(1)); pc.pose.thresh_win_step = int(v.at(2));
    } else if (a == "--min-perim") pc.pose.min_marker_perimeter_rate = std::stod(next());
    else if (a == "--ball-downscale") pc.detector.downscale = std::stoi(next());
    else if (a == "--median") pc.detector.median_kernel = std::stoi(next());
    else if (a == "--blur") pc.detector.blur_kernel = std::stoi(next());
    else if (a == "--marker") pc.pose.marker_length_m = std::stod(next());
    else if (a == "--ids") {
      auto v = parse_list(next());
      for (int k = 0; k < 4; ++k) pc.pose.corner_ids[k] = int(v.at(k));
    } else if (a == "--pos") {
      auto v = parse_list(next());
      for (int k = 0; k < 4; ++k) pc.pose.corner_pos_m[k] = {v.at(2 * k), v.at(2 * k + 1)};
    } else if (a == "--sweep") sweep = true;
    else if (a == "--image") image_path = next();
    else if (a == "--iters") iters = std::stoi(next());
    else { std::fprintf(stderr, "Argumento desconocido: %s\n", a.c_str()); return 2; }
  }

  if (sweep) {
    struct C { const char* fcc; int w, h; };
    const C combos[] = {{"YUYV", 640, 480}, {"MJPG", 640, 480}, {"YUYV", 1280, 720},
                        {"MJPG", 1280, 720}, {"MJPG", 320, 240}};
    std::printf("%-22s %-46s %7s %10s %8s\n", "pedido", "obtenido", "Hz", "jitter ms", "max ms");
    for (const auto& c : combos) {
      CameraConfig k = cc;
      k.fourcc = c.fcc; k.width = c.w; k.height = c.h;
      try {
        auto r = capture_only(k, seconds);
        char req[32];
        std::snprintf(req, sizeof req, "%s %dx%d@%.0f", c.fcc, c.w, c.h, cc.fps);
        std::printf("%-22s %-46s %7.1f %10.2f %8.1f\n", req, r.got.substr(0, 46).c_str(), r.hz,
                    r.jitter, r.max_ms);
      } catch (const std::exception& e) {
        std::printf("%s %dx%d: %s\n", c.fcc, c.w, c.h, e.what());
      }
    }
    return 0;
  }

  Intrinsics intr;
  try { intr = Intrinsics::load(intr_path); }
  catch (...) { std::printf("(sin intrinsecos, usando aproximados)\n"); intr = Intrinsics::guess(cc.width, cc.height); }
  VisionPipeline pipe(pc, intr);

  if (!image_path.empty()) {
    const cv::Mat img = cv::imread(image_path);
    if (img.empty()) { std::fprintf(stderr, "No se pudo leer %s\n", image_path.c_str()); return 2; }
    std::vector<double> total, aruco, ball;
    VisionResult r;
    for (int i = 0; i < iters; ++i) {
      const double t0 = now_s();
      r = pipe.process(img, i / 30.0);
      total.push_back((now_s() - t0) * 1e3);
      aruco.push_back(r.aruco_ms);
      ball.push_back(r.ball_ms);
    }
    std::printf("C++ | %d iter | total mediana %.3f ms p95 %.3f ms | aruco %.3f ms | pelota %.3f ms"
                " | pose %s (%.3f, %.3f) grados pelota %s (%.1f, %.1f) cm\n",
                iters, pct(total, 50), pct(total, 95), pct(aruco, 50), pct(ball, 50),
                r.pose.valid ? "ok" : "NO", r.pose.theta_x * 57.29578, r.pose.theta_y * 57.29578, r.ball.valid ? "ok" : "NO", r.ball.x * 100, r.ball.y * 100);
    return 0;
  }

  Camera cam(cc);
  std::printf("Camara: %s\n", cam.actual_settings().c_str());
  cam.start();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  RateStats cam_rate(1000000), proc(1000000);
  std::vector<double> aruco, ball, total, latency;
  long last = -1, n = 0, dropped = 0, pose_ok = 0, ball_ok = 0;
  Frame f;
  const double end = now_s() + seconds;
  while (now_s() < end) {
    if (!cam.wait_next(last, f)) continue;
    if (last >= 0) dropped += f.seq - last - 1;
    last = f.seq;
    const double t0 = now_s();
    auto r = pipe.process(f.image, f.t_mono);
    const double t1 = now_s();
    total.push_back((t1 - t0) * 1e3);
    latency.push_back((t1 - f.t_mono) * 1e3);
    proc.tick(t1);
    aruco.push_back(r.aruco_ms);
    ball.push_back(r.ball_ms);
    pose_ok += r.pose.valid;
    ball_ok += r.ball.valid;
    ++n;
  }
  const long read_total = cam.frames_read();
  cam.stop();

  auto p = proc.summary();
  const double cam_hz = (n + dropped) / seconds;
  std::printf("\nCuadros procesados: %ld  descartados: %ld  (leidos en total %ld)\n", n, dropped, read_total);
  std::printf("Camara entrega:   ~%6.1f Hz\n", cam_hz);
  std::printf("Pipeline procesa:  %6.1f Hz  jitter %.2f ms  max %.1f ms\n", p.hz, p.jitter_ms, p.period_max_ms);
  std::printf("ArUco    mediana %5.2f ms  p95 %5.2f ms\n", pct(aruco, 50), pct(aruco, 95));
  std::printf("Pelota   mediana %5.2f ms  p95 %5.2f ms\n", pct(ball, 50), pct(ball, 95));
  std::printf("Total    mediana %5.2f ms  p95 %5.2f ms\n", pct(total, 50), pct(total, 95));
  std::printf("Latencia captura->resultado mediana %5.2f ms  p95 %5.2f ms\n", pct(latency, 50), pct(latency, 95));
  std::printf("Pose valida %.0f%%   pelota valida %.0f%%\n", 100.0 * pose_ok / std::max(n, 1L),
              100.0 * ball_ok / std::max(n, 1L));
  const double budget = 1e3 / cc.fps;
  std::printf("\nPresupuesto a %.0f FPS: %.1f ms/cuadro -> %s\n", cc.fps, budget,
              pct(total, 95) < budget ? "CABE" : "NO CABE (p95 excede)");
  return 0;
}

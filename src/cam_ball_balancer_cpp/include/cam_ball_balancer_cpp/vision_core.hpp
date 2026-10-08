// Nucleo de vision en C++ (sin ROS): port 1:1 de la version Python
// (ball_tracker.py, kalman.py, platform_pose.py, vision_pipeline.py).
// Las decisiones de diseno estan documentadas alla; aqui solo se anotan
// las diferencias propias del port.
#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/aruco.hpp>

namespace cbb {

// ---------------------------------------------------------------- Kalman
struct KalmanConfig {
  double process_noise_pos = 1.0e-2;
  double process_noise_vel = 20.0;
  double measurement_noise = 5.0e-3;
  double initial_pos_var = 1.0e-2;
  double initial_vel_var = 1.0;
};

// Estado [x, y, vx, vy], mediciones de posicion. Matrices fijas 4x4
// (cv::Matx) -> sin allocaciones por cuadro.
class Kalman2D {
 public:
  explicit Kalman2D(KalmanConfig cfg = {});
  void reset(double x, double y);
  void predict(double dt);
  void update(double zx, double zy);
  bool initialized = false;
  double px() const { return x_(0); }
  double py() const { return x_(1); }

 private:
  KalmanConfig cfg_;
  cv::Vec4d x_;
  cv::Matx44d P_;
};

// ---------------------------------------------------------------- Pelota
struct BallDetectorConfig {
  cv::Scalar hsv_lower{5, 120, 120};
  cv::Scalar hsv_upper{18, 255, 255};
  int median_kernel = 5;
  int blur_kernel = 7;
  int morph_kernel = 5;
  double min_radius_px = 4.0;
  double max_radius_px = 200.0;
  double min_circularity = 0.6;
  // Detectar a 1/downscale de resolucion (medido en la Pi: la pelota a
  // resolucion completa costaba ~10 ms/cuadro). Con la pelota de ~25-30 px
  // de radio, a la mitad sobra resolucion; el centro vuelve a escala completa.
  int downscale = 2;
};

struct Detection {
  double u, v, r;
};

class BallDetector {
 public:
  explicit BallDetector(BallDetectorConfig cfg = {});
  // Busca en `frame` (ya recortado al ROI si aplica). Buffers reutilizados.
  std::optional<Detection> detect(const cv::Mat& frame);
  BallDetectorConfig cfg;

 private:
  cv::Mat small_, denoised_, blurred_, hsv_, mask_, kernel_;
};

enum class TrackState { Searching, Tracking, Coasting, Lost, NoPose };
const char* to_string(TrackState s);

struct BallMeasurement {
  double x = std::nan(""), y = std::nan("");
  double t = 0;
  bool valid = false;
  std::optional<cv::Point2d> pixel;
  double radius_px = 0;
  TrackState state = TrackState::Searching;
};

// ------------------------------------------------------ Camara / plato
struct Intrinsics {
  cv::Matx33d K;
  cv::Mat dist;  // 1xN
  static Intrinsics load(const std::string& yml_path);  // cv::FileStorage
  static Intrinsics guess(int w, int h);
};

struct PoseConfig {
  int dictionary = cv::aruco::DICT_4X4_50;
  double marker_length_m = 0.03;
  // Tags en las esquinas: orden TL, TR, BR, BL (como se ven en la imagen).
  // Posiciones en el frame del plato (REP-103): origen al centro, X a la
  // derecha, Y hacia ARRIBA en la imagen, Z saliendo del plato hacia la
  // camara. Por eso TL tiene y > 0.
  std::array<int, 4> corner_ids{0, 1, 2, 3};
  std::array<cv::Point2d, 4> corner_pos_m{
      cv::Point2d{-0.09, 0.09}, {0.09, 0.09}, {0.09, -0.09}, {-0.09, -0.09}};
  // Ambiguedad planar de IPPE: cerca de "plato de frente a la camara" hay
  // dos soluciones casi igual de buenas (la real y su espejo). Si
  // err_espejo / err_mejor < ambiguity_ratio, se elige la mas cercana a la
  // pose anterior en vez de la de menor error (evita saltos +a / -a).
  double ambiguity_ratio = 2.0;
  // Cuadros seguidos sin pose antes de olvidar la pose anterior.
  int max_frames_without_pose = 10;
  // Por encima de este error (px RMS) la solucion IPPE se considera
  // degenerada y se recalcula con SQPnP.
  double max_reproj_err_px = 20.0;
  // Costo de ArUco (Pi3): detectar sobre la imagen reducida `aruco_downscale` veces y refinar las esquinas
  // con cornerSubPix a resolucion completa; ventanas del umbral adaptativo (cada paso es una pasada
  // completa sobre la imagen: 3,23,10 = 3 pasadas); perimetro minimo relativo de un tag (descarta
  // contornos chicos antes de decodificar).
  int aruco_downscale = 1;
  int thresh_win_min = 3, thresh_win_max = 23, thresh_win_step = 10;
  double min_marker_perimeter_rate = 0.03;
};

struct PlatePose {
  bool valid = false;
  double reproj_err_px = 0;   // RMS de la solucion elegida
  double ambiguity = 0;       // err_otra / err_mejor (cerca de 1 = ambiguo)
  bool disambiguated = false; // true si se eligio por continuidad temporal
  double theta_x = 0, theta_y = 0;
  cv::Vec3d rvec, tvec;
  std::vector<cv::Point2f> corner_px;  // centros de los tags vistos
};

class PoseEstimator {
 public:
  PoseEstimator(PoseConfig cfg, Intrinsics intr);
  // `search` = sub-rectangulo donde buscar los tags (vacio = todo el cuadro).
  PlatePose estimate(const cv::Mat& gray, const cv::Rect& search = {});
  PoseConfig cfg;

 private:
  Intrinsics intr_;
  cv::Ptr<cv::aruco::Dictionary> dict_;
  cv::Ptr<cv::aruco::DetectorParameters> params_;
  std::vector<std::vector<cv::Point2f>> corners_, rejected_;
  cv::Mat small_;
  std::vector<int> ids_;
  std::optional<cv::Vec3d> prev_rvec_;
  int frames_without_pose_ = 0;
};

void tilt_from_rvec(const cv::Vec3d& rvec, double& theta_x, double& theta_y);

// Pixel -> (x, y) en el frame del plato, en el plano z = +ball_radius
// (centro de la pelota, sobre el plato hacia la camara).
class PlateRayMapper {
 public:
  PlateRayMapper(Intrinsics intr, double ball_radius_m);
  void set_pose(const cv::Vec3d& rvec, const cv::Vec3d& tvec);
  bool ready() const { return ready_; }
  bool to_plane(double u, double v, double& x, double& y) const;

 private:
  Intrinsics intr_;
  double r_;
  cv::Matx33d R_;
  cv::Vec3d t_;
  bool ready_ = false;
};

// ------------------------------------------------------------ Tracker
class BallTracker {
 public:
  BallTracker(BallDetectorConfig det, KalmanConfig kf, int max_misses,
              double gate_max_speed, double gate_min_jump);
  // `roi` en coordenadas del cuadro completo (vacio = todo el cuadro).
  BallMeasurement update(const cv::Mat& frame, double t, const cv::Rect& roi,
                         const PlateRayMapper& mapper);
  BallDetector detector;
  TrackState state = TrackState::Searching;

 private:
  Kalman2D kf_;
  int max_misses_;
  double gate_max_speed_, gate_min_jump_;
  std::optional<double> last_t_;
  int misses_ = 0;
};

// ------------------------------------------------------------ Tasa
class RateStats {
 public:
  explicit RateStats(size_t window = 120) : window_(window) {}
  void tick(double t_s);
  struct Summary { double hz = 0, period_ms = 0, jitter_ms = 0, period_max_ms = 0; };
  Summary summary() const;
  void reset();

 private:
  size_t window_;
  std::deque<double> periods_;
  std::optional<double> last_;
  mutable std::mutex mtx_;
};

// ----------------------------------------------------------- Pipeline
struct PipelineConfig {
  BallDetectorConfig detector;
  KalmanConfig kalman;
  PoseConfig pose;
  int aruco_every_n = 1;
  int roi_margin_px = 30;
  // ArUco busca primero solo alrededor del plato del cuadro anterior (ROI +
  // este margen extra por si el plato se movio); si falla, cuadro completo.
  int aruco_roi_margin_px = 40;
  bool use_roi = true;
  double ball_radius_m = 0.02;
  int max_consecutive_misses = 15;
  double gate_max_speed = 2.0;  // m/s
  double gate_min_jump = 0.02;  // m
  int pose_hold_max_frames = 10;
};

struct VisionResult {
  double t = 0;
  PlatePose pose;
  bool pose_fresh = false;
  BallMeasurement ball;
  double aruco_ms = 0, ball_ms = 0;
  cv::Rect roi;
};

class VisionPipeline {
 public:
  VisionPipeline(PipelineConfig cfg, Intrinsics intr);
  VisionResult process(const cv::Mat& bgr, double t);
  const PipelineConfig cfg;

 private:
  void update_roi(const PlatePose& p, cv::Size size);
  Intrinsics intr_;
  PoseEstimator pose_est_;
  PlateRayMapper mapper_;
  BallTracker tracker_;
  cv::Mat gray_;
  cv::Rect roi_;
  std::optional<PlatePose> last_pose_;
  int pose_age_ = 0;
  long frame_idx_ = 0;
};

cv::Mat draw_debug(const cv::Mat& bgr, const VisionResult& r, double hz);

inline double now_s() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

}  // namespace cbb

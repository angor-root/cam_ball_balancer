// vision_node (C++): camara + pose del plato + pelota en un proceso.
// Hacia ROS solo salen datos (misma interfaz que la version Python):
//   ball_position, platform_pose, vision/stats, plate_marker,
//   debug_image/compressed (solo si debug_image_hz > 0).
// Todos los mensajes de un cuadro llevan el stamp de captura.
//
// Arbol TF (REP-103, Z hacia arriba):
//   world --(estatico: camara a camera_height_m, mirando hacia abajo)--> camera_link
//   camera_link --(estatico: rotacion estandar)--> camera_optical_frame   (z adelante, x der, y abajo)
//   camera_optical_frame --(medido, ArUco)--> plate_link   (Z saliendo del plato hacia la camara)
//   plate_link --(medido)--> ball_link
// `world` = plato plano de referencia: con el plato sin inclinar, plate_link
// queda paralelo a world. Fixed Frame en RViz: world.
#include <atomic>
#include <limits>
#include <memory>
#include <thread>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker.hpp>

#include <cam_ball_balancer_msgs/msg/ball_position.hpp>
#include <cam_ball_balancer_msgs/msg/platform_pose.hpp>
#include <cam_ball_balancer_msgs/msg/vision_stats.hpp>

#include "cam_ball_balancer_cpp/camera.hpp"
#include "cam_ball_balancer_cpp/vision_core.hpp"

using cam_ball_balancer_msgs::msg::BallPosition;
using cam_ball_balancer_msgs::msg::PlatformPose;
using cam_ball_balancer_msgs::msg::VisionStats;
using geometry_msgs::msg::TransformStamped;

namespace {

std::string default_intrinsics() {
  try {
    return ament_index_cpp::get_package_share_directory("cam_ball_balancer_cpp") +
           "/config/camera_intrinsics.yml";
  } catch (...) {
    return "";
  }
}

void quat_from_rvec(const cv::Vec3d& rvec, double& x, double& y, double& z, double& w) {
  const double a = cv::norm(rvec);
  if (a < 1e-12) { x = y = z = 0; w = 1; return; }
  const cv::Vec3d k = rvec / a;
  const double s = std::sin(a / 2);
  x = k[0] * s; y = k[1] * s; z = k[2] * s; w = std::cos(a / 2);
}

}  // namespace

class VisionNode : public rclcpp::Node {
 public:
  VisionNode() : Node("vision_node") {
    cbb::CameraConfig cc;
    cc.device = declare_parameter("device", std::string("0"));
    cc.width = declare_parameter("width", 640);
    cc.height = declare_parameter("height", 480);
    cc.fps = declare_parameter("fps", 30.0);
    cc.fourcc = declare_parameter("fourcc", std::string("MJPG"));
    cc.exposure = declare_parameter("exposure", -1.0);
    cc.wb_temperature = declare_parameter("wb_temperature", -1.0);
    const auto intr_path = declare_parameter("camera_intrinsics_path", default_intrinsics());

    world_frame_ = declare_parameter("world_frame_id", std::string("world"));
    camera_link_frame_ = declare_parameter("camera_link_frame_id", std::string("camera_link"));
    camera_frame_ = declare_parameter("camera_frame_id", std::string("camera_optical_frame"));
    camera_height_ = declare_parameter("camera_height_m", 0.30);
    const bool publish_static = declare_parameter("publish_camera_static_tf", true);
    plate_frame_ = declare_parameter("plate_frame_id", std::string("plate_link"));
    ball_frame_ = declare_parameter("ball_frame_id", std::string("ball_link"));
    const bool publish_tf = declare_parameter("publish_tf", true);

    cbb::PipelineConfig pc;
    pc.pose.marker_length_m = declare_parameter("marker_length_m", 0.03);
    const auto ids = declare_parameter("corner_marker_ids", std::vector<int64_t>{0, 1, 2, 3});
    const auto pos = declare_parameter(
        "corner_positions_m", std::vector<double>{-0.09, 0.09, 0.09, 0.09, 0.09, -0.09, -0.09, -0.09});
    if (ids.size() != 4 || pos.size() != 8)
      throw std::invalid_argument("corner_marker_ids debe tener 4 y corner_positions_m 8 valores");
    for (int i = 0; i < 4; ++i) {
      pc.pose.corner_ids[i] = int(ids[i]);
      pc.pose.corner_pos_m[i] = {pos[2 * i], pos[2 * i + 1]};
    }
    pc.aruco_every_n = declare_parameter("aruco_every_n", 1);
    pc.pose.ambiguity_ratio = declare_parameter("ambiguity_ratio", 2.0);
    pc.pose.aruco_downscale = declare_parameter("aruco_downscale", 1);
    pc.pose.thresh_win_min = declare_parameter("aruco_thresh_win_min", 3);
    pc.pose.thresh_win_max = declare_parameter("aruco_thresh_win_max", 23);
    pc.pose.thresh_win_step = declare_parameter("aruco_thresh_win_step", 10);
    pc.pose.min_marker_perimeter_rate = declare_parameter("aruco_min_perimeter_rate", 0.03);
    pc.pose_hold_max_frames = declare_parameter("pose_hold_max_frames", 10);
    pc.pose.max_frames_without_pose = pc.pose_hold_max_frames;
    const auto lo = declare_parameter("hsv_lower", std::vector<int64_t>{5, 120, 120});
    const auto hi = declare_parameter("hsv_upper", std::vector<int64_t>{18, 255, 255});
    pc.detector.hsv_lower = cv::Scalar(double(lo[0]), double(lo[1]), double(lo[2]));
    pc.detector.hsv_upper = cv::Scalar(double(hi[0]), double(hi[1]), double(hi[2]));
    pc.detector.min_radius_px = declare_parameter("min_radius_px", 4.0);
    pc.detector.max_radius_px = declare_parameter("max_radius_px", 200.0);
    pc.ball_radius_m = declare_parameter("ball_radius_m", 0.02);
    pc.use_roi = declare_parameter("use_roi", true);
    pc.roi_margin_px = declare_parameter("roi_margin_px", 30);
    pc.aruco_roi_margin_px = declare_parameter("aruco_roi_margin_px", 40);
    pc.detector.downscale = declare_parameter("ball_downscale", 2);
    pc.detector.median_kernel = declare_parameter("ball_median_kernel", 5);
    pc.detector.blur_kernel = declare_parameter("ball_blur_kernel", 7);
    pc.max_consecutive_misses = declare_parameter("max_consecutive_misses", 15);
    pc.gate_max_speed = declare_parameter("gate_max_speed", 2.0);
    pc.gate_min_jump = declare_parameter("gate_min_jump", 0.02);
    const double debug_hz = declare_parameter("debug_image_hz", 0.0);
    const double stats_hz = declare_parameter("stats_hz", 2.0);

    cbb::Intrinsics intr;
    try {
      intr = cbb::Intrinsics::load(intr_path);
      RCLCPP_INFO(get_logger(), "Intrinsecos: %s", intr_path.c_str());
    } catch (const std::exception& e) {
      intr = cbb::Intrinsics::guess(cc.width, cc.height);
      RCLCPP_WARN(get_logger(), "%s -> usando intrinsecos aproximados (angulos/metros aproximados)",
                  e.what());
    }
    pipeline_ = std::make_unique<cbb::VisionPipeline>(pc, intr);

    pub_ball_ = create_publisher<BallPosition>("ball_position", 10);
    pub_pose_ = create_publisher<PlatformPose>("platform_pose", 10);
    pub_stats_ = create_publisher<VisionStats>("vision/stats", 10);
    pub_plate_marker_ = create_publisher<visualization_msgs::msg::Marker>("plate_marker", 10);
    if (publish_tf) tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    if (publish_static) publish_static_camera_tf();
    // Tamano del plato para el marcador: caja que envuelve los tags.
    for (const auto& c : pc.pose.corner_pos_m) {
      plate_half_x_ = std::max(plate_half_x_, std::abs(c.x) + pc.pose.marker_length_m / 2);
      plate_half_y_ = std::max(plate_half_y_, std::abs(c.y) + pc.pose.marker_length_m / 2);
    }
    if (debug_hz > 0) {
      debug_period_ = 1.0 / debug_hz;
      pub_debug_ = create_publisher<sensor_msgs::msg::CompressedImage>("debug_image/compressed", 1);
    }
    stats_period_ = 1.0 / std::max(stats_hz, 0.1);

    camera_ = std::make_unique<cbb::Camera>(cc);
    RCLCPP_INFO(get_logger(), "Camara (lo que acepto el driver): %s", camera_->actual_settings().c_str());
    camera_->start();
    worker_ = std::thread(&VisionNode::loop, this);
  }

  ~VisionNode() override {
    running_ = false;
    if (worker_.joinable()) worker_.join();
    camera_->stop();
  }

 private:
  void loop() {
    long last_seq = -1;
    double last_stats = 0, last_debug = 0;
    cbb::Frame fr;
    while (running_ && rclcpp::ok()) {
      if (!camera_->wait_next(last_seq, fr, 1.0)) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Sin cuadros de la camara en 1 s");
        continue;
      }
      if (last_seq >= 0 && fr.seq > last_seq + 1) dropped_ += fr.seq - last_seq - 1;
      last_seq = fr.seq;

      const auto res = pipeline_->process(fr.image, fr.t_mono);

      const double tp = cbb::now_s();
      const rclcpp::Time stamp(fr.t_wall_ns, RCL_SYSTEM_TIME);
      publish_data(res, stamp);
      const double publish_ms = (cbb::now_s() - tp) * 1e3;

      const double now = cbb::now_s();
      proc_rate_.tick(now);
      if (now - last_stats >= stats_period_) {
        last_stats = now;
        publish_stats(res, stamp, publish_ms, (now - fr.t_mono) * 1e3);
      }
      if (pub_debug_ && now - last_debug >= debug_period_) {
        last_debug = now;
        publish_debug(fr.image, res, stamp);
      }
    }
  }

  void publish_static_camera_tf() {
    // Rotaciones como matrices (columnas = ejes del hijo en el padre).
    // world -> optico: camara mirando hacia abajo, imagen "arriba" = +Y world:
    //   x_opt = +X_w, y_opt = -Y_w, z_opt = -Z_w
    const cv::Matx33d R_w_opt(1, 0, 0, 0, -1, 0, 0, 0, -1);
    // camera_link -> optico (estandar REP-103): x_opt = -y_l, y_opt = -z_l, z_opt = +x_l
    const cv::Matx33d R_l_opt(0, 0, 1, -1, 0, 0, 0, -1, 0);
    const cv::Matx33d R_w_l = R_w_opt * R_l_opt.t();

    auto make = [&](const std::string& parent, const std::string& child, const cv::Matx33d& R,
                    const cv::Vec3d& t) {
      TransformStamped tf;
      tf.header.stamp = now();
      tf.header.frame_id = parent;
      tf.child_frame_id = child;
      tf.transform.translation.x = t[0];
      tf.transform.translation.y = t[1];
      tf.transform.translation.z = t[2];
      cv::Vec3d rv;
      cv::Rodrigues(R, rv);
      auto& q = tf.transform.rotation;
      quat_from_rvec(rv, q.x, q.y, q.z, q.w);
      return tf;
    };
    static_tf_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);
    static_tf_->sendTransform({
        make(world_frame_, camera_link_frame_, R_w_l, {0, 0, camera_height_}),
        make(camera_link_frame_, camera_frame_, R_l_opt, {0, 0, 0}),
    });
  }

  void publish_data(const cbb::VisionResult& r, const rclcpp::Time& stamp) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    PlatformPose pm;
    pm.header.stamp = stamp;
    pm.header.frame_id = camera_frame_;
    // Invalida => NaN (no 0: 0 deg se confundiria con "plato plano").
    pm.theta_x = r.pose.valid ? r.pose.theta_x : nan;
    pm.theta_y = r.pose.valid ? r.pose.theta_y : nan;
    pm.tx = r.pose.valid ? r.pose.tvec[0] : nan;
    pm.ty = r.pose.valid ? r.pose.tvec[1] : nan;
    pm.tz = r.pose.valid ? r.pose.tvec[2] : nan;
    pm.valid = r.pose.valid;
    pub_pose_->publish(pm);

    BallPosition bm;
    bm.header.stamp = stamp;
    bm.header.frame_id = plate_frame_;
    bm.x = r.ball.x;
    bm.y = r.ball.y;
    bm.valid = r.ball.valid;
    pub_ball_->publish(bm);

    if (!tf_) return;
    std::vector<TransformStamped> tfs;
    if (r.pose.valid) {
      TransformStamped t;
      t.header.stamp = stamp;
      t.header.frame_id = camera_frame_;
      t.child_frame_id = plate_frame_;
      t.transform.translation.x = r.pose.tvec[0];
      t.transform.translation.y = r.pose.tvec[1];
      t.transform.translation.z = r.pose.tvec[2];
      auto& q = t.transform.rotation;
      quat_from_rvec(r.pose.rvec, q.x, q.y, q.z, q.w);
      tfs.push_back(t);

      visualization_msgs::msg::Marker mk;
      mk.header.stamp = stamp;
      mk.header.frame_id = plate_frame_;
      mk.ns = "plate";
      mk.type = visualization_msgs::msg::Marker::CUBE;
      mk.action = visualization_msgs::msg::Marker::ADD;
      mk.pose.position.z = -0.002;
      mk.pose.orientation.w = 1.0;
      mk.scale.x = 2 * plate_half_x_;
      mk.scale.y = 2 * plate_half_y_;
      mk.scale.z = 0.004;
      mk.color.r = 0.2f; mk.color.g = 0.6f; mk.color.b = 1.0f; mk.color.a = 0.6f;
      mk.lifetime = rclcpp::Duration::from_seconds(0.5);
      pub_plate_marker_->publish(mk);
    }
    if (r.ball.valid) {
      TransformStamped t;
      t.header.stamp = stamp;
      t.header.frame_id = plate_frame_;
      t.child_frame_id = ball_frame_;
      t.transform.translation.x = r.ball.x;
      t.transform.translation.y = r.ball.y;
      t.transform.translation.z = pipeline_->cfg.ball_radius_m;  // sobre el plato
      t.transform.rotation.w = 1.0;
      tfs.push_back(t);
    }
    if (!tfs.empty()) tf_->sendTransform(tfs);
  }

  void publish_stats(const cbb::VisionResult& r, const rclcpp::Time& stamp, double publish_ms,
                     double latency_ms) {
    const auto s = proc_rate_.summary();
    VisionStats m;
    m.header.stamp = stamp;
    m.header.frame_id = camera_frame_;
    m.rate_hz = s.hz;
    m.period_ms = s.period_ms;
    m.jitter_ms = s.jitter_ms;
    m.period_max_ms = s.period_max_ms;
    m.camera_hz = camera_->rate.summary().hz;
    m.aruco_ms = r.aruco_ms;
    m.ball_ms = r.ball_ms;
    m.publish_ms = publish_ms;
    m.latency_ms = latency_ms;
    m.dropped_frames = dropped_;
    pub_stats_->publish(m);
  }

  void publish_debug(const cv::Mat& img, const cbb::VisionResult& r, const rclcpp::Time& stamp) {
    sensor_msgs::msg::CompressedImage msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = camera_frame_;
    msg.format = "jpeg";
    cv::imencode(".jpg", cbb::draw_debug(img, r, proc_rate_.summary().hz), msg.data,
                 {cv::IMWRITE_JPEG_QUALITY, 70});
    pub_debug_->publish(msg);
  }

  std::unique_ptr<cbb::VisionPipeline> pipeline_;
  std::unique_ptr<cbb::Camera> camera_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_plate_marker_;
  std::string world_frame_, camera_link_frame_;
  double camera_height_ = 0.30, plate_half_x_ = 0, plate_half_y_ = 0;
  rclcpp::Publisher<BallPosition>::SharedPtr pub_ball_;
  rclcpp::Publisher<PlatformPose>::SharedPtr pub_pose_;
  rclcpp::Publisher<VisionStats>::SharedPtr pub_stats_;
  rclcpp::Publisher<sensor_msgs::msg::CompressedImage>::SharedPtr pub_debug_;
  std::string camera_frame_, plate_frame_, ball_frame_;
  double debug_period_ = 0, stats_period_ = 0.5;
  cbb::RateStats proc_rate_;
  uint32_t dropped_ = 0;
  std::atomic<bool> running_{true};
  std::thread worker_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VisionNode>();
  rclcpp::spin(node);
  node.reset();
  rclcpp::shutdown();
  return 0;
}

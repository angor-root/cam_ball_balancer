# Ball position on the plate, as returned by the vision pipeline's
# obtener_posicion_pelota() contract.
std_msgs/Header header

# Position in the plate frame, meters, origin at plate center, same
# convention as PlatformPose below. NaN when valid == false.
float64 x
float64 y

# False when the ball has not been detected recently enough to trust
# the (Kalman-predicted) position — consumers should hold their last
# safe command / stop, not act on x, y.
bool valid



def imu_orientation_estimate_robust_kalman(time: list[float], accel_data: list[tuple[float, float, float]], gyro_data: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """
    Estimates the orientation of an IMU using a robust adaptive error state Kalman filter.
    Algorithm is taken from https://www.sciencedirect.com/science/article/pii/S0263224124019821?via%3Dihub.

    Args:
        time: Timestamps of the IMU measurements.
        accel_data: Accelerometer measurements as (x, y, z).
        gyro_data: Gyroscope measurements as (x, y, z).

    Returns:
        Estimated orientation as (roll, pitch, yaw) at each timestamp.
    """
    return


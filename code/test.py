import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import factorized


# ============================================================
# USER OPTIONS
# ============================================================

MOTION_MODE = "axial"
# "axial", "45deg", "transverse", "custom"

CUSTOM_ANGLE_DEG = 30.0

ACCEL_PEAK = 8.0             # [m/s^2]
ARM_SWING_FREQUENCY = 1.8    # [Hz]

T_SIM = 5.0                  # [s]

# ------------------------------------------------------------
# Cepstral filtering
# ------------------------------------------------------------

CEPSTRAL_STOP_HALF_WIDTH = 0.12   # [s]

# Also suppress the period associated with the acceleration's
# second harmonic:
FILTER_SECOND_HARMONIC = True


# ============================================================
# Vessel / blood parameters
# ============================================================

D = 2.7e-3                   # Vessel diameter [m]
R = D / 2

mu = 3e-3                    # Blood viscosity [Pa*s]
rho = 1060.0                 # Blood density [kg/m^3]
nu = mu / rho


# ============================================================
# Heart parameters
# ============================================================

heart_rate = 70.0            # [BPM]
T_beat = 60.0 / heart_rate

target_mean_velocity = 0.15  # [m/s]


# ============================================================
# Numerical parameters
# ============================================================

dt = 0.002                   # 500 Hz
fs = 1 / dt

N_r = 61

r = np.linspace(
    0,
    R,
    N_r
)

dr = r[1] - r[0]


# ============================================================
# Acceleration direction
# ============================================================

if MOTION_MODE == "axial":

    acceleration_angle = 0.0

elif MOTION_MODE == "45deg":

    acceleration_angle = 45.0

elif MOTION_MODE == "transverse":

    acceleration_angle = 90.0

elif MOTION_MODE == "custom":

    acceleration_angle = CUSTOM_ANGLE_DEG

else:

    raise ValueError(
        "Unknown MOTION_MODE"
    )


theta = np.deg2rad(
    acceleration_angle
)


# ============================================================
# Vessel acceleration
# ============================================================

def vessel_acceleration(t):

    fundamental = np.sin(
        2
        * np.pi
        * ARM_SWING_FREQUENCY
        * t
    )

    second_harmonic = (
        0.20
        * np.sin(
            2
            * np.pi
            * (2 * ARM_SWING_FREQUENCY)
            * t
            + 0.4
        )
    )

    waveform = (
        fundamental
        + second_harmonic
    )

    waveform /= 1.15

    return (
        ACCEL_PEAK
        * waveform
    )


def acceleration_components(t):

    a = vessel_acceleration(t)

    a_axial = (
        a * np.cos(theta)
    )

    a_transverse = (
        a * np.sin(theta)
    )

    return (
        a_axial,
        a_transverse
    )


# ============================================================
# Cardiac pressure waveform
# ============================================================

def pulse_shape(phase):

    systolic_peak = (
        1.8
        * np.exp(
            -0.5
            * (
                (phase - 0.18)
                / 0.065
            ) ** 2
        )
    )

    dicrotic_notch = (
        -0.18
        * np.exp(
            -0.5
            * (
                (phase - 0.36)
                / 0.025
            ) ** 2
        )
    )

    reflected_wave = (
        0.28
        * np.exp(
            -0.5
            * (
                (phase - 0.48)
                / 0.09
            ) ** 2
        )
    )

    baseline = 0.25

    return (
        baseline
        + systolic_peak
        + dicrotic_notch
        + reflected_wave
    )


# Normalize waveform
phase_test = np.linspace(
    0,
    1,
    10000,
    endpoint=False
)

pulse_mean = np.mean(
    pulse_shape(
        phase_test
    )
)


# Mean pressure gradient corresponding
# approximately to desired mean velocity
G_mean = (
    8
    * mu
    * target_mean_velocity
    / R**2
)


def cardiac_pressure_gradient(t):

    phase = (
        (t % T_beat)
        / T_beat
    )

    return (
        G_mean
        * pulse_shape(phase)
        / pulse_mean
    )


# ============================================================
# Pressure-gradient functions
# ============================================================

def pressure_gradient_no_motion(t):

    return (
        cardiac_pressure_gradient(t)
    )


def pressure_gradient_with_motion(t):

    G_heart = (
        cardiac_pressure_gradient(t)
    )

    a_axial, _ = (
        acceleration_components(t)
    )

    # Apparent pressure-gradient contribution
    # from axial acceleration
    return (
        G_heart
        - rho * a_axial
    )


# ============================================================
# Radial Navier-Stokes operator
# ============================================================

M = N_r - 1

L = lil_matrix(
    (M, M)
)


# Centerline symmetry
L[0, 0] = (
    -4 / dr**2
)

L[0, 1] = (
    4 / dr**2
)


for i in range(
    1,
    M
):

    ri = r[i]

    left = (
        1 / dr**2
        - 1 / (
            2 * ri * dr
        )
    )

    center = (
        -2 / dr**2
    )

    right = (
        1 / dr**2
        + 1 / (
            2 * ri * dr
        )
    )

    L[i, i - 1] = left

    L[i, i] = center

    if i + 1 < M:

        L[i, i + 1] = right


L = L.tocsc()


# ============================================================
# Crank-Nicolson solver
# ============================================================

I = eye(
    M,
    format="csc"
)

A = (
    I
    - 0.5
    * dt
    * nu
    * L
)

B = (
    I
    + 0.5
    * dt
    * nu
    * L
)


solve = factorized(A)

forcing_vector = np.ones(M)


# ============================================================
# Precompute integration weights
# ============================================================

integration_weights = np.ones(
    N_r
)

integration_weights[0] = 0.5
integration_weights[-1] = 0.5


mean_weights = (
    2
    / R**2
    * dr
    * integration_weights[:-1]
    * r[:-1]
)


# ============================================================
# Flow simulation
# ============================================================

def simulate_flow(
    pressure_gradient_function
):

    u = np.zeros(M)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    warmup_time = 5.0

    current_time = (
        -warmup_time
    )

    G0 = (
        pressure_gradient_function(
            current_time
        )
    )

    n_warmup = int(
        warmup_time
        / dt
    )

    for _ in range(
        n_warmup
    ):

        new_time = (
            current_time
            + dt
        )

        G1 = (
            pressure_gradient_function(
                new_time
            )
        )

        forcing = (
            0.5
            * dt
            * (G0 + G1)
            / rho
            * forcing_vector
        )

        u = solve(
            B @ u
            + forcing
        )

        current_time = (
            new_time
        )

        G0 = G1


    # --------------------------------------------------------
    # Main simulation
    # --------------------------------------------------------

    # endpoint=False gives an even number
    # of samples here: 2500 samples
    time = np.arange(
        0,
        T_SIM,
        dt
    )

    mean_velocity = (
        np.zeros_like(time)
    )

    G0 = (
        pressure_gradient_function(
            0.0
        )
    )

    for k, t in enumerate(
        time
    ):

        mean_velocity[k] = (
            np.dot(
                mean_weights,
                u
            )
        )

        if (
            k
            == len(time) - 1
        ):

            break

        G1 = (
            pressure_gradient_function(
                t + dt
            )
        )

        forcing = (
            0.5
            * dt
            * (G0 + G1)
            / rho
            * forcing_vector
        )

        u = solve(
            B @ u
            + forcing
        )

        G0 = G1


    return (
        time,
        mean_velocity
    )


# ============================================================
# COMPLEX CEPSTRUM
# ============================================================

def complex_cepstrum(x):

    x = np.asarray(
        x,
        dtype=float
    )

    N = len(x)

    # Fourier transform
    X = np.fft.fft(x)

    magnitude = np.abs(X)

    # Prevent log(0)
    eps = (
        np.finfo(float).eps
        * max(
            1.0,
            np.max(magnitude)
        )
    )

    # --------------------------------------------------------
    # Phase unwrapping
    # --------------------------------------------------------

    phase = np.unwrap(
        np.angle(X)
    )

    # FFT bin corresponding approximately to pi
    center = (
        (N + 1)
        // 2
    )

    # Linear phase / circular-delay correction
    ndelay = int(
        np.round(
            phase[center]
            / np.pi
        )
    )

    phase_corrected = (
        phase
        - np.pi
        * ndelay
        * np.arange(N)
        / center
    )

    # --------------------------------------------------------
    # Complex logarithm
    # --------------------------------------------------------

    log_spectrum = (
        np.log(
            np.maximum(
                magnitude,
                eps
            )
        )
        + 1j
        * phase_corrected
    )

    # Complex cepstrum
    cepstrum = np.real(
        np.fft.ifft(
            log_spectrum
        )
    )

    return (
        cepstrum,
        ndelay
    )


# ============================================================
# INVERSE COMPLEX CEPSTRUM
# ============================================================

def inverse_complex_cepstrum(
    cepstrum,
    ndelay
):

    cepstrum = np.asarray(
        cepstrum,
        dtype=float
    )

    N = len(
        cepstrum
    )

    log_spectrum = np.fft.fft(
        cepstrum
    )

    center = (
        (N + 1)
        // 2
    )

    # Restore removed linear phase
    phase = (
        np.imag(
            log_spectrum
        )
        + np.pi
        * ndelay
        * np.arange(N)
        / center
    )

    # Reconstruct complex spectrum
    spectrum = np.exp(
        np.real(
            log_spectrum
        )
        + 1j
        * phase
    )

    # Back to time domain
    x = np.real(
        np.fft.ifft(
            spectrum
        )
    )

    return x


# ============================================================
# CEPSTRAL BAND-STOP / LIFTER
# ============================================================

def cepstral_bandstop(
    cepstrum,
    fs,
    stop_periods,
    half_width
):

    N = len(
        cepstrum
    )

    n = np.arange(N)

    # Signed quefrency:
    #
    #   0 ... +T
    #   then negative quefrencies
    #
    quefrency_signed = np.where(
        n <= N // 2,
        n / fs,
        (n - N) / fs
    )

    lifter = np.ones(N)

    for period in stop_periods:

        stop_region = (
            np.abs(
                np.abs(
                    quefrency_signed
                )
                - period
            )
            <= half_width
        )

        lifter[
            stop_region
        ] = 0.0


    filtered_cepstrum = (
        cepstrum
        * lifter
    )

    return (
        filtered_cepstrum,
        lifter,
        quefrency_signed
    )


# ============================================================
# Run simulations
# ============================================================

print(
    "Simulating blood flow..."
)

time, velocity_true = (
    simulate_flow(
        pressure_gradient_no_motion
    )
)

_, velocity_motion = (
    simulate_flow(
        pressure_gradient_with_motion
    )
)


# ============================================================
# Acceleration
# ============================================================

axial_acceleration = np.array(
    [
        acceleration_components(t)[0]
        for t in time
    ]
)


# ============================================================
# Known periods from acceleration
# ============================================================

arm_period = (
    1
    / ARM_SWING_FREQUENCY
)

second_harmonic_period = (
    1
    / (
        2
        * ARM_SWING_FREQUENCY
    )
)


stop_periods = [
    arm_period
]


if FILTER_SECOND_HARMONIC:

    stop_periods.append(
        second_harmonic_period
    )


# ============================================================
# Complex cepstra
# ============================================================

cep_true, delay_true = (
    complex_cepstrum(
        velocity_true
    )
)

cep_motion, delay_motion = (
    complex_cepstrum(
        velocity_motion
    )
)

cep_acceleration, delay_accel = (
    complex_cepstrum(
        axial_acceleration
    )
)


# ============================================================
# Apply cepstral band-stop
# ============================================================

(
    cep_motion_filtered,
    lifter,
    q_signed
) = cepstral_bandstop(
    cep_motion,
    fs,
    stop_periods,
    CEPSTRAL_STOP_HALF_WIDTH
)


# ============================================================
# Reconstruct blood-flow signal
# ============================================================

velocity_reconstructed = (
    inverse_complex_cepstrum(
        cep_motion_filtered,
        delay_motion
    )
)


# ============================================================
# Sanity check:
# inverse without filtering should reproduce original
# ============================================================

velocity_identity = (
    inverse_complex_cepstrum(
        cep_motion,
        delay_motion
    )
)

identity_error = np.sqrt(
    np.mean(
        (
            velocity_identity
            - velocity_motion
        ) ** 2
    )
)


# ============================================================
# Positive quefrency axis for plots
# ============================================================

N = len(time)

quefrency = (
    np.arange(N)
    / fs
)


q_min = 0.1
q_max = 1.5


q_mask = (
    (quefrency >= q_min)
    & (quefrency <= q_max)
)


# ============================================================
# ERROR METRICS
# ============================================================

rmse_corrupted = np.sqrt(
    np.mean(
        (
            velocity_motion
            - velocity_true
        ) ** 2
    )
)


rmse_reconstructed = np.sqrt(
    np.mean(
        (
            velocity_reconstructed
            - velocity_true
        ) ** 2
    )
)


# ============================================================
# MAIN PLOTS
# ============================================================

fig, (
    ax1,
    ax2,
    ax3,
    ax4,
    ax5
) = plt.subplots(
    5,
    1,
    figsize=(12, 16)
)


# ------------------------------------------------------------
# 1. Time-domain blood flow
# ------------------------------------------------------------

ax1.plot(
    time,
    velocity_true,
    label="Underlying blood flow",
    linewidth=2.5
)

ax1.plot(
    time,
    velocity_motion,
    label="Blood flow with motion",
    linewidth=1.2,
    alpha=0.6
)

ax1.plot(
    time,
    velocity_reconstructed,
    "--",
    label="After cepstral band-stop",
    linewidth=2
)

ax1.set_xlabel(
    "Time [s]"
)

ax1.set_ylabel(
    "Mean velocity [m/s]"
)

ax1.set_title(
    "Blood-flow reconstruction using complex-cepstrum filtering"
)

ax1.grid(
    True,
    alpha=0.3
)

ax1.legend()


# ------------------------------------------------------------
# 2. Complex cepstrum — underlying blood
# ------------------------------------------------------------

ax2.plot(
    quefrency[q_mask],
    cep_true[q_mask],
    linewidth=2
)

ax2.axvline(
    T_beat,
    linestyle="--",
    label=f"Heart period ({T_beat:.3f} s)"
)

ax2.axvline(
    arm_period,
    linestyle=":",
    label=f"Arm period ({arm_period:.3f} s)"
)

if FILTER_SECOND_HARMONIC:

    ax2.axvline(
        second_harmonic_period,
        linestyle=":",
        alpha=0.6,
        label=(
            "2nd harmonic period "
            f"({second_harmonic_period:.3f} s)"
        )
    )

ax2.set_xlabel(
    "Quefrency [s]"
)

ax2.set_ylabel(
    "Cepstral amplitude"
)

ax2.set_title(
    "Complex cepstrum — underlying blood flow"
)

ax2.grid(
    True,
    alpha=0.3
)

ax2.legend()


# ------------------------------------------------------------
# 3. Complex cepstrum — corrupted blood
# ------------------------------------------------------------

ax3.plot(
    quefrency[q_mask],
    cep_motion[q_mask],
    linewidth=2
)

ax3.axvline(
    T_beat,
    linestyle="--",
    label=f"Heart period ({T_beat:.3f} s)"
)

ax3.axvline(
    arm_period,
    linestyle=":",
    label=f"Arm period ({arm_period:.3f} s)"
)

if FILTER_SECOND_HARMONIC:

    ax3.axvline(
        second_harmonic_period,
        linestyle=":",
        alpha=0.6,
        label=(
            "2nd harmonic period "
            f"({second_harmonic_period:.3f} s)"
        )
    )

ax3.set_xlabel(
    "Quefrency [s]"
)

ax3.set_ylabel(
    "Cepstral amplitude"
)

ax3.set_title(
    "Complex cepstrum — blood flow with motion"
)

ax3.grid(
    True,
    alpha=0.3
)

ax3.legend()


# ------------------------------------------------------------
# 4. Complex cepstrum — after cepstral band-stop
# ------------------------------------------------------------

ax4.plot(
    quefrency[q_mask],
    cep_motion_filtered[q_mask],
    linewidth=2
)

ax4.axvline(
    T_beat,
    linestyle="--",
    label=f"Heart period ({T_beat:.3f} s)"
)

ax4.axvline(
    arm_period,
    linestyle=":",
    label=f"Removed arm period ({arm_period:.3f} s)"
)

if FILTER_SECOND_HARMONIC:

    ax4.axvline(
        second_harmonic_period,
        linestyle=":",
        alpha=0.6,
        label=(
            "Removed 2nd harmonic "
            f"({second_harmonic_period:.3f} s)"
        )
    )

ax4.set_xlabel(
    "Quefrency [s]"
)

ax4.set_ylabel(
    "Cepstral amplitude"
)

ax4.set_title(
    "Complex cepstrum — after motion-period suppression"
)

ax4.grid(
    True,
    alpha=0.3
)

ax4.legend()


# ------------------------------------------------------------
# 5. Complex cepstrum — acceleration
# ------------------------------------------------------------

ax5.plot(
    quefrency[q_mask],
    cep_acceleration[q_mask],
    linewidth=2
)

ax5.axvline(
    arm_period,
    linestyle=":",
    label=f"Arm period ({arm_period:.3f} s)"
)

if FILTER_SECOND_HARMONIC:

    ax5.axvline(
        second_harmonic_period,
        linestyle=":",
        alpha=0.6,
        label=(
            "2nd harmonic period "
            f"({second_harmonic_period:.3f} s)"
        )
    )

ax5.set_xlabel(
    "Quefrency [s]"
)

ax5.set_ylabel(
    "Cepstral amplitude"
)

ax5.set_title(
    "Complex cepstrum — axial acceleration"
)

ax5.grid(
    True,
    alpha=0.3
)

ax5.legend()


plt.tight_layout()
plt.show()


# ============================================================
# Plot the cepstral "band-stop filter" itself
# ============================================================

positive_q = (
    q_signed >= 0
)

plot_lifter = (
    positive_q
    & (q_signed <= q_max)
)


plt.figure(
    figsize=(10, 4)
)

plt.plot(
    q_signed[plot_lifter],
    lifter[plot_lifter],
    linewidth=2
)

plt.axvline(
    arm_period,
    linestyle="--",
    label=f"Arm period ({arm_period:.3f} s)"
)

if FILTER_SECOND_HARMONIC:

    plt.axvline(
        second_harmonic_period,
        linestyle="--",
        label=(
            "2nd harmonic period "
            f"({second_harmonic_period:.3f} s)"
        )
    )

plt.ylim(
    -0.1,
    1.1
)

plt.xlim(
    0,
    q_max
)

plt.xlabel(
    "Quefrency [s]"
)

plt.ylabel(
    "Lifter gain"
)

plt.title(
    "Cepstral band-stop / lifter"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# Print results
# ============================================================

print()

print(
    f"Heart period: "
    f"{T_beat:.4f} s"
)

print(
    f"Arm period: "
    f"{arm_period:.4f} s"
)

print(
    f"Second harmonic period: "
    f"{second_harmonic_period:.4f} s"
)

print()

print(
    f"Complex cepstrum inverse sanity-check RMSE: "
    f"{identity_error:.3e} m/s"
)

print()

print(
    f"RMSE before cepstral filtering: "
    f"{rmse_corrupted:.4f} m/s"
)

print(
    f"RMSE after cepstral filtering: "
    f"{rmse_reconstructed:.4f} m/s"
)
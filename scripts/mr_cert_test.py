import sys
import os
import math
import platform
import time
from datetime import datetime
import secrets
from multiprocessing import Pool, cpu_count, current_process

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

# Some Python/OS combinations re-execute this module's top-level code once per
# worker process when the Pool starts (e.g. under the "spawn" start method).
# Gate the startup banner to the main process only so it doesn't get reprinted.
_IS_MAIN_PROCESS = current_process().name == 'MainProcess'

try:
    import gmpy2
    USE_GMP = True
    if _IS_MAIN_PROCESS:
        print(f"[SUCCESS] gmpy2 loaded successfully! C-GMP Engine Active (v{gmpy2.mp_version()}).")
except ImportError as e:
    USE_GMP = False
    if _IS_MAIN_PROCESS:
        print(f"[WARNING] gmpy2 NOT FOUND: {e}")

FIRST_200_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331,
    337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457,
    461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599,
    601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
    739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877,
    881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019,
    1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123,
    1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223
]

# Config: how many rounds per stage.
K1_DEFAULT = 100
K2_DEFAULT = 100

# How many logical cores to leave free for other work on the machine by default.
# Change this if you want a different default headroom.
RESERVED_CORES_DEFAULT = 2


def get_available_core_count():
    """
    Number of logical cores this process could actually use.
    Prefers sched_getaffinity (respects taskset/cgroup CPU pinning, e.g. if you've
    already restricted this shell or a container to a subset of cores) and falls
    back to the total system core count if that's not available (e.g. on Windows/macOS).
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return cpu_count()


def choose_worker_count():
    """
    Figure out how many worker processes to use, defaulting to something that
    leaves headroom for other workloads on the machine, and let the user
    override it interactively.
    """
    total_available = get_available_core_count()
    suggested = max(1, total_available - RESERVED_CORES_DEFAULT)

    print(f"[!] {total_available} logical cores available to this process "
          f"(reserving {RESERVED_CORES_DEFAULT} by default for other work).")
    raw = input(f"    Worker processes to use [{suggested}]: ").strip()

    if raw == "":
        return suggested
    try:
        chosen = int(raw)
        if chosen < 1:
            print("    Invalid value, must be >= 1. Using default.")
            return suggested
        if chosen > total_available:
            print(f"    Warning: {chosen} exceeds {total_available} available cores - "
                  f"you may see thrashing/oversubscription.")
        return chosen
    except ValueError:
        print("    Couldn't parse that, using default.")
        return suggested


def get_hardware_info():
    info = {'os': f"{platform.system()} {platform.release()} ({platform.architecture()})", 'cpu': "Unknown CPU", 'ram': "Unknown RAM"}
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line or "Model" in line:
                        info['cpu'] = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Windows":
            info['cpu'] = platform.processor()
    except Exception:
        pass
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        info['ram'] = f"{round(kb / (1024**2), 2)} GB"
                        break
    except Exception:
        pass
    return info


def calculate_probabilities(k_total, k1, k2, prime1, prime2):
    def to_sci(k):
        log_p = -k * math.log10(4)
        return f"{10 ** (log_p - math.floor(log_p)):.4f} x 10^{math.floor(log_p)}"
    return (to_sci(k1) if prime1 else "1.0000", to_sci(k2) if prime2 else "1.0000",
            to_sci(k_total) if (prime1 and prime2) else "1.0000")


def format_etr(seconds_remaining):
    if seconds_remaining < 60:
        return f"{seconds_remaining:.1f} sec"
    minutes = int(seconds_remaining // 60)
    seconds = int(seconds_remaining % 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    return f"{hours}h {minutes}m"


# ---- Worker: runs in a separate process. Must be top-level (picklable) and
# must re-import gmpy2 itself, since mpz objects don't cross process boundaries.
def _mr_round_worker(args):
    a_int, n_str, d_str, s = args
    import gmpy2 as g
    n_gmp = g.mpz(n_str)
    d_gmp = g.mpz(d_str)
    n_minus_1 = n_gmp - 1
    a_gmp = g.mpz(a_int)

    x = g.powmod(a_gmp, d_gmp, n_gmp)
    if x == 1 or x == n_minus_1:
        return (a_int, True)
    for _ in range(s - 1):
        x = g.powmod(x, 2, n_gmp)
        if x == n_minus_1:
            return (a_int, True)
    return (a_int, False)


def generate_bases(stage_mode, k_iterations, n, global_used_set):
    bases = []
    while len(bases) < k_iterations:
        if stage_mode == "primes":
            a_int = FIRST_200_PRIMES[len(bases)]
        else:
            a_int = secrets.randbits(64)
        if a_int in global_used_set or a_int >= n:
            continue
        global_used_set.add(a_int)
        bases.append(a_int)
    return bases


def run_single_mr_test_parallel(n, stage_mode, k_iterations, d, s, stage_name,
                                 global_used_set, pool, n_str, d_str):
    bases = generate_bases(stage_mode, k_iterations, n, global_used_set)

    start_stage_time = time.perf_counter()
    completed = 0
    all_passed = True
    witness = None

    work_items = [(a, n_str, d_str, s) for a in bases]

    # imap_unordered lets us print progress as results land, same spirit as
    # the original's incremental progress lines, without waiting for the
    # slowest worker to hold up the printout.
    for a_int, passed in pool.imap_unordered(_mr_round_worker, work_items):
        completed += 1
        if not passed:
            all_passed = False
            witness = a_int
            # Don't break early: workers already dispatched will keep running,
            # but we stop waiting on new progress prints once we know it's composite.

        if completed % 10 == 0 or completed == k_iterations:
            elapsed = time.perf_counter() - start_stage_time
            avg_time = elapsed / completed
            remaining_iters = k_iterations - completed
            print(f"   -> {stage_name}: {completed}/{k_iterations} iterations complete. "
                  f"[Ave. {avg_time:.4f} sec/iter | ETR: {format_etr(remaining_iters * avg_time)}]")

    return all_passed, bases


def main():
    print("=== Certified Miller-Rabin Dual-Stage Primality Verifier (Parallel) ===")
    if USE_GMP:
        print(f"[!] gmpy2 active (GMP v{gmpy2.mp_version()}): Fast C-GMP Core Enabled.")
    else:
        print("[!] WARNING: Running without gmpy2 - this script requires it for the worker processes.")
        sys.exit(1)

    n_workers = choose_worker_count()
    print(f"[!] Using {n_workers} worker process(es) for modexp rounds.\n")

    expr_input = input("Enter expression to test (e.g., 7**20176 - 2):\n> ")
    expr_clean = expr_input.strip()

    print("\nEvaluating expression...")
    n = eval(expr_clean)
    k1, k2 = K1_DEFAULT, K2_DEFAULT
    k_total = k1 + k2

    fn_expr = expr_clean.replace("**", "^").replace(" ", "")

    digit_length = int(gmpy2.log10(gmpy2.mpz(n))) + 1

    print(f"\n[1/4] Gathering system info...")
    hw = get_hardware_info()

    print("[2/4] Factoring exponent sequence...")
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1

    # n and d are passed to workers as strings (via mpz round-trip on the
    # worker side) since raw huge Python ints pickle fine but this keeps the
    # IPC payload format explicit and avoids relying on fork semantics.
    n_str = str(n)
    d_str = str(d)

    global_used_set = set()

    # Lower this process's (and its workers') scheduling priority so the OS
    # favors your other running work when both want the CPU at the same time.
    # Workers inherit niceness from the parent on fork, so setting it once
    # here before the Pool starts covers all of them. Windows has no os.nice.
    if hasattr(os, "nice"):
        try:
            os.nice(10)
        except Exception:
            pass

    with Pool(processes=n_workers) as pool:
        print(f"[3/4] Launching Test Run 1 (First {k1} consecutive primes) across {n_workers} workers...")
        start1 = datetime.now()
        prime1, bases1 = run_single_mr_test_parallel(
            n, "primes", k1, d, s, "Stage 1", global_used_set, pool, n_str, d_str)
        end1 = datetime.now()

        if not prime1:
            final_verdict = "DEFINITIVELY COMPOSITE"
            prime2, bases2 = False, []
            start2 = end2 = datetime.now()
            print(f"\n[RESULT] Stage 1 found a witness proving compositeness - skipping Stage 2.")
        else:
            print(f"[4/4] Launching Test Run 2 ({k2} random 64-bit bases) across {n_workers} workers...")
            start2 = datetime.now()
            prime2, bases2 = run_single_mr_test_parallel(
                n, "random", k2, d, s, "Stage 2", global_used_set, pool, n_str, d_str)
            end2 = datetime.now()
            final_verdict = "PROBABLE PRIME (PRP)" if prime2 else "DEFINITIVELY COMPOSITE"

    print(f"\n{'=' * 60}")
    print(f"  VERDICT: {final_verdict}")
    print(f"{'=' * 60}\n")

    p1_str, p2_str, comb_str = calculate_probabilities(k_total, k1, k2, prime1, prime2)

    cert_filename = f"MR_{fn_expr}_certification.txt"
    with open(cert_filename, "w", encoding="utf-8") as f:
        f.write("================================================================================\n")
        f.write("               PRIMALITY TEST VERIFICATION & CERTIFICATE RECORD                 \n")
        f.write("================================================================================\n\n")
        f.write("--- TARGET SPECIFICATIONS ---\n")
        f.write(f"Expression Evaluated       : {expr_clean}\n")
        f.write(f"Total Base-10 Integer Width: {digit_length} digits\n")
        f.write(f"Math Engine Engine Core    : gmpy2 (C-GMP Assembly, {n_workers}-way parallel)\n\n")
        f.write("--- VERIFICATION METRICS ---\n")
        f.write(f"Final Combined Verdict     : {final_verdict}\n")
        f.write(f"Total Unique Bases Checked : {k_total}\n")
        f.write(f"False-Positive Error Limit : {comb_str}\n\n")
        f.write("--- HARDWARE ENVIRONMENT SIGNATURE ---\n")
        f.write(f"CPU Model                  : {hw['cpu']}\n")
        f.write(f"System RAM Capacity        : {hw['ram']}\n")
        f.write(f"Host Operating System      : {hw['os']}\n")
        f.write(f"Parallel Worker Processes  : {n_workers}\n\n")
        f.write("--- STAGE 1 LOG EXECUTION ---\n")
        f.write(f"Iterations Assigned        : {k1} rounds\n")
        f.write(f"Execution Started          : {start1.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write(f"Execution Completed        : {end1.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write(f"Stage 1 Result             : {'Passed' if prime1 else 'Failed'}\n")
        f.write(f"Stage 1 Error Bound        : {p1_str}\n\n")
        f.write("--- STAGE 2 LOG EXECUTION ---\n")
        f.write(f"Iterations Assigned        : {k2} rounds\n")
        f.write(f"Execution Started          : {start2.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write(f"Execution Completed        : {end2.strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write(f"Stage 2 Result             : {'Passed' if prime2 else 'Failed'}\n")
        f.write(f"Stage 2 Error Bound        : {p2_str}\n\n")
        f.write("================================================================================\n")
        f.write("                       FULL TRACE AUDIT: BASES ENUMERATED                       \n")
        f.write("================================================================================\n")
        f.write("\n[STAGE 1 WITNESSES]\n")
        for index, b in enumerate(bases1, 1):
            f.write(f"Base 1.{index:03d}: {b}\n")
        f.write("\n[STAGE 2 WITNESSES]\n")
        for index, b in enumerate(bases2, 1):
            f.write(f"Base 2.{index:03d}: {b}\n")

    print(f"\nDone! Certificate record generated: '{cert_filename}'")


if __name__ == "__main__":
    main()

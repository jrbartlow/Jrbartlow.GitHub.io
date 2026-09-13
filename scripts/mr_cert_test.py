import sys
import math
import platform
import subprocess
import time
from datetime import datetime
import secrets
import json
import os

# FORCE PYTHON TO ALLOW UNLIMITED INTEGER STRINGS GLOBALLY
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

# Attempt to load high-speed GMP library
try:
    import gmpy2
    USE_GMP = True
except ImportError:
    USE_GMP = False

def get_hardware_info():
    """Gathers CPU model, total RAM, and OS specifications optimized for Linux environments."""
    info = {'os': f"{platform.system()} {platform.release()} ({platform.architecture()})", 'cpu': "Unknown CPU", 'ram': "Unknown RAM"}
    
    # 1. Fixed Linux CPU Model extraction
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line or "Model" in line:
                        # Correctly split on colon, take the second half, and strip strings
                        info['cpu'] = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Windows":
            info['cpu'] = platform.processor()
        elif platform.system() == "Darwin":
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            info['cpu'] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()
    except Exception:
        pass

    # 2. Fixed Linux RAM Capacity extraction
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        # Split columns safely and pull the middle numerical value
                        kb = int(line.split()[1])
                        info['ram'] = f"{round(kb / (1024**2), 2)} GB"
                        break
        elif platform.system() == "Windows":
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info['ram'] = f"{round(stat.ullTotalPhys / (1024**3), 2)} GB"
        elif platform.system() == "Darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip().decode()
            info['ram'] = f"{round(int(out) / (1024**3), 2)} GB"
    except Exception:
        pass
        
    return info

def calculate_probabilities(k_total, k1, k2, prime1, prime2):
    def to_sci(k):
        log_p = -k * math.log10(4)
        return f"{10 ** (log_p - math.floor(log_p)):.4f} x 10^{math.floor(log_p)}"
    return (to_sci(k1) if prime1 else "1.0000", to_sci(k2) if prime2 else "1.0000", to_sci(k_total) if (prime1 and prime2) else "1.0000")

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

def run_single_mr_test(n, k_iterations, d, s, stage_name, global_used_set, recovery_file=None):
    """Runs a Miller-Rabin test with live dynamic updates and fault-tolerant crash recovery."""
    used_bases = []
    start_stage_time = time.perf_counter()
    
    if USE_GMP:
        n_gmp = gmpy2.mpz(n)
        d_gmp = gmpy2.mpz(d)
        n_minus_1 = n_gmp - 1
    else:
        n_gmp, d_gmp, n_minus_1 = n, d, n - 1

    # Check for an existing crash recovery ledger file on disk
    if recovery_file and os.path.exists(recovery_file):
        try:
            with open(recovery_file, "r") as rf:
                recovery_data = json.load(rf)
                if stage_name in recovery_data:
                    # Reload pre-validated bases from previous session
                    loaded_bases = recovery_data[stage_name]
                    print(f"   [+] Recovery Ledger Found: Resuming {stage_name} from iteration {len(loaded_bases)}...")
                    for b in loaded_bases:
                        global_used_set.add(b)
                        used_bases.append(b)
        except Exception as e:
            print(f"   [-] Warning reading recovery file (Starting fresh): {e}")

    # Set temporary print interval for the calibration round
    print_interval = 1 
    first_iter_timed = False

    while len(used_bases) < k_iterations:
        iter_start = time.perf_counter()
        
        a_int = secrets.randbelow(int(n) - 4) + 2 
        if a_int in global_used_set:
            continue
            
        global_used_set.add(a_int)
        used_bases.append(a_int)
        
        # Core Miller-Rabin modular squaring arithmetic
        if USE_GMP:
            x = gmpy2.powmod(gmpy2.mpz(a_int), d_gmp, n_gmp)
            if x != 1 and x != n_minus_1:
                for _ in range(s - 1):
                    x = gmpy2.powmod(x, 2, n_gmp)
                    if x == n_minus_1:
                        break
                else:
                    return False, used_bases
        else:
            x = pow(a_int, d, n)
            if x != 1 and x != n - 1:
                for _ in range(s - 1):
                    x = pow(x, 2, n)
                    if x == n - 1:
                        break
                else:
                    return False, used_bases

        # Dynamic Calibration Step on the very first completed iteration
        iter_duration = time.perf_counter() - iter_start
        if not first_iter_timed:
            # If a single iteration takes more than 10 minutes (600s), print every iteration
            if iter_duration > 600:
                print_interval = 1
                print(f"   [!] Calibration Warning: Iteration time ({iter_duration:.1f}s) exceeds 10 minutes.")
                print(f"       Enabling every-iteration status logging and crash recovery ledger.")
            else:
                print_interval = 10
            first_iter_timed = True

        current_count = len(used_bases)

        # Write to recovery log file ONLY if running a long-term (10+ min per base) workload
        if recovery_file and print_interval == 1:
            try:
                # Read existing file content or start empty dictionary structures
                state = {}
                if os.path.exists(recovery_file):
                    with open(recovery_file, "r") as rf:
                        state = json.load(rf)
                state[stage_name] = used_bases
                with open(recovery_file, "w") as wf:
                    json.dump(state, wf)
            except Exception:
                pass # Fail silently during high-speed writes to prevent interrupting math core

        # Console Progress Tracking Milestone Check
        if current_count % print_interval == 0 or current_count == k_iterations:
            elapsed = time.perf_counter() - start_stage_time
            avg_time = elapsed / current_count
            remaining_iters = k_iterations - current_count
            print(f"   -> {stage_name}: {current_count}/{k_iterations} iterations complete. [Ave. {avg_time:.3f} sec/iter | ETR: {format_etr(remaining_iters * avg_time)}]")
            
    # Clean up recovery ledger upon successful completion of the full test stage
    if recovery_file and os.path.exists(recovery_file) and stage_name == "Stage 2":
        try:
            os.remove(recovery_file)
        except Exception:
            pass

    return True, used_bases

def main():
    print("=== Certified Miller-Rabin Dual-Stage Primality Verifier ===")
    if USE_GMP:
        print("[!] gmpy2 active: Using high-speed GNU Multiple Precision Arithmetic core.")
    else:
        print("[!] gmpy2 NOT found: Falling back to pure Python software multiplication engine.")

    expr_input = input("\nEnter the number or math expression to test (e.g., 7**274120 - 2):\n> ")
    expr_clean = expr_input.strip()
    if not all(c in "0123456789+-*/() \t" for c in expr_clean):
        print("Error: Invalid characters.")
        return
        
    print("\nEvaluating expression...")
    try:
        n = eval(expr_clean)
    except Exception as e:
        print(f"Error: {e}")
        return

    try:
        k_total = int(input("\nEnter total iterations (k) to split (e.g., 100):\n> "))
        if k_total < 2: raise ValueError("k must be >= 2.")
    except ValueError as e:
        print(f"Invalid input: {e}")
        return

    k1 = k_total // 2
    k2 = k_total - k1
    fn_expr = expr_clean.replace("**", "^").replace(" ", "")
    recovery_filename = f"MR_{fn_expr}_recovery.json"

    if n <= 1 or n % 2 == 0 or n <= 3:
        print(f"Result: {n} evaluated immediately out of testing range.")
        return

    # High-speed logarithm base digit counting (Zero risk of string limits crashing)
    if USE_GMP:
        digit_length = int(gmpy2.log10(gmpy2.mpz(n))) + 1
    else:
        digit_length = math.floor(math.log10(n)) + 1

    print(f"\n[1/4] Gathering system architecture configuration...")
    hw = get_hardware_info()

    print("[2/4] Factoring exponent sequence...")
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1

    global_used_set = set()

    # Stage 1 Execution
    print(f"[3/4] Launching Test Run 1 ({k1} iterations)...")
    start1 = datetime.now()
    prime1, bases1 = run_single_mr_test(n, k1, d, s, "Stage 1", global_used_set, recovery_file=recovery_filename)
    end1 = datetime.now()

    # Stage 2 Execution (Only runs if Stage 1 passed)
    if not prime1:
        final_verdict = "DEFINITIVELY COMPOSITE"
        prime2, bases2 = False, []
        start2 = end2 = datetime.now()
    else:
        print(f"[4/4] Launching Test Run 2 ({k2} iterations)...")
        start2 = datetime.now()
        prime2, bases2 = run_single_mr_test(n, k2, d, s, "Stage 2", global_used_set, recovery_file=recovery_filename)
        end2 = datetime.now()
        final_verdict = "PROBABLE PRIME (PRP)" if prime2 else "DEFINITIVELY COMPOSITE"

    p1_str, p2_str, comb_str = calculate_probabilities(k_total, k1, k2, prime1, prime2)

    cert_filename = f"MR_{fn_expr}_certification.txt"
    try:
        with open(cert_filename, "w", encoding="utf-8") as f:
            f.write("================================================================================\n")
            f.write("               PRIMALITY TEST VERIFICATION & CERTIFICATE RECORD                 \n")
            f.write("================================================================================\n\n")
            f.write("--- TARGET SPECIFICATIONS ---\n")
            f.write(f"Expression Evaluated       : {expr_clean}\n")
            f.write(f"Total Base-10 Integer Width: {digit_length} digits\n")
            f.write(f"Math Engine Engine Core    : {'gmpy2 (C-GMP Assembly)' if USE_GMP else 'Pure Python (Software Math)'}\n\n")
            f.write("--- VERIFICATION METRICS ---\n")
            f.write(f"Final Combined Verdict     : {final_verdict}\n")
            f.write(f"Total Unique Bases Checked : {k_total}\n")
            f.write(f"False-Positive Error Limit : {comb_str}\n\n")
            f.write("--- HARDWARE ENVIRONMENT SIGNATURE ---\n")
            f.write(f"CPU Model                  : {hw['cpu']}\n")
            f.write(f"System RAM Capacity        : {hw['ram']}\n")
            f.write(f"Host Operating System      : {hw['os']}\n\n")
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
            for index, b in enumerate(bases1, 1): f.write(f"Base 1.{index:03d}: {b}\n")
            f.write("\n[STAGE 2 WITNESSES]\n")
            for index, b in enumerate(bases2, 1): f.write(f"Base 2.{index:03d}: {b}\n")
        print(f"\nSuccess! Generated certification record file: '{cert_filename}'")
    except Exception as e:
        print(f"Error generating certification text file: {e}")
        
if __name__ == "__main__":
    main()


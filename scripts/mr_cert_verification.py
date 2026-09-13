import sys
import re
import time
import platform
import subprocess
import json
import os

# FORCE PYTHON TO ALLOW UNLIMITED INTEGER STRINGS GLOBALLY
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

try:
    import gmpy2
    USE_GMP = True
except ImportError:
    USE_GMP = False

def get_hardware_info():
    """Gathers verifier's CPU model, total RAM, and OS specifications across Linux, Windows, and Mac."""
    info = {'os': f"{platform.system()} {platform.release()} ({platform.architecture()})", 'cpu': "Unknown CPU", 'ram': "Unknown RAM"}
    
    # Cross-platform CPU detection
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line or "Model" in line:
                        info['cpu'] = line.split(":", 1)[1].strip()
                        break
        elif platform.system() == "Windows":
            info['cpu'] = platform.processor()
        elif platform.system() == "Darwin":
            import os
            os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
            info['cpu'] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).strip().decode()
    except Exception:
        pass

    # Cross-platform RAM detection
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
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

def parse_certification_file(filename):
    expression = None
    bases = []
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    expr_match = re.search(r"Expression Evaluated\s*:\s*(.+)", content)
    if expr_match:
        expression = expr_match.group(1).strip()
    base_matches = re.findall(r"Base \d+\.\d+:\s*(\d+)", content)
    for b in base_matches:
        bases.append(int(b))
    return expression, bases

def format_etr(seconds_remaining):
    """Converts seconds into a clean, human-readable format."""
    if seconds_remaining < 60:
        return f"{seconds_remaining:.1f} sec"
    minutes = int(seconds_remaining // 60)
    seconds = int(seconds_remaining % 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours = int(minutes // 60)
    minutes = int(minutes % 60)
    return f"{hours}h {minutes}m"

def verify_certificate(cert_filename):
    print(f"=== Certificate Verification Audit Tool ===")
    if USE_GMP:
        print("[!] gmpy2 active: Verification math accelerated via GMP library.")
    else:
        print("[!] gmpy2 NOT found: Using fallback standard python evaluation blocks.")
        
    print(f"Reading target file: '{cert_filename}'...\n")
    try:
        expression, bases = parse_certification_file(cert_filename)
    except Exception as e:
        print(f"[-] Error reading or parsing the certificate file: {e}")
        return

    if not expression or not bases:
        print("[-] Error: Could not extract target data fields from certificate.")
        return
    
    # Prompt the user to choose an audit mode
    print("Select Verification Mode:")
    print("  [1] Full Audit  (Verify all bases in the certificate - Maximum Certainty)")
    print("  [2] Quick Audit (Verify 10 randomly sampled bases - High Speed)")
    mode_choice = input("Select option (1 or 2, default is 1): > ").strip()
    
    audit_type = "FULL AUDIT"
    if mode_choice == "2":
        import random
        audit_type = "QUICK AUDIT (10 Randomly Sampled Bases)"
        # Randomly sample 10 unique bases from the file if there are enough
        if len(bases) > 10:
            bases = random.sample(bases, 10)

    print(f"[1/3] Parsing verification target expressions...")
    try:
        n = eval(expression)
    except Exception as e:
        print(f"[-] Error evaluating expression: {e}")
        return
        
    print(f"[2/3] Pre-calculating exponentiation factors...")
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2; s += 1
        
    print(f"[3/3] Commencing verification processing loop...")
    verifier_hw = get_hardware_info()
    verified = True
    
    # Establish recovery file tokens based on the certificate target name
    fn_expr = expression.replace("**", "^").replace(" ", "")
    recovery_file = f"MR_{fn_expr}_verify_recovery.json"
    
    # Structure to hold indexes of fully audited bases
    verified_indices = []
    start_idx = 0

    # Look for an existing crash recovery ledger file on disk
    if os.path.exists(recovery_file):
        try:
            with open(recovery_file, "r") as rf:
                recovery_data = json.load(rf)
                if "verified_bases_count" in recovery_data:
                    start_idx = recovery_data["verified_bases_count"]
                    print(f"   [+] Recovery Ledger Found: Resuming verification from base index {start_idx}...")
        except Exception as e:
            print(f"   [-] Warning reading recovery file (Starting fresh): {e}")

    # Initialize calculation timers and dynamic interval defaults
    start_calc_time = time.perf_counter()
    k_total = len(bases)
    print_interval = 1
    first_base_timed = False

    # Slice the bases list to pick up exactly where a previous interrupted run left off
    bases_to_test = bases[start_idx:]

    # Cast core target variables to high-speed GMP strings if active
    if USE_GMP:
        n_gmp = gmpy2.mpz(n)
        d_gmp = gmpy2.mpz(d)
        n_minus_1 = n_gmp - 1

    # Execute the sequential verification loop
    for current_offset, a in enumerate(bases_to_test, 1):
        actual_global_idx = start_idx + current_offset
        base_start = time.perf_counter()

        # Execute Modular Exponentiation
        if USE_GMP:
            x = gmpy2.powmod(gmpy2.mpz(a), d_gmp, n_gmp)
            if x != 1 and x != n_minus_1:
                for _ in range(s - 1):
                    x = gmpy2.powmod(x, 2, n_gmp)
                    if x == n_minus_1:
                        break
                else:
                    print(f"\n[!] VERIFICATION FAILED at base trace sequence element #{actual_global_idx}!")
                    verified = False
                    break
        else:
            x = pow(a, d, n)
            if x != 1 and x != n - 1:
                for _ in range(s - 1):
                    x = pow(x, 2, n)
                    if x == n - 1:
                        break
                else:
                    print(f"\n[!] VERIFICATION FAILED at base trace sequence element #{actual_global_idx}!")
                    verified = False
                    break

        # Dynamic Calibration Step on the very first analyzed base element
        base_duration = time.perf_counter() - base_start
        if not first_base_timed:
            if base_duration > 600:  # Exceeds 10 minutes
                print_interval = 1
                print(f"   [!] Calibration Note: Base analysis time ({base_duration:.1f}s) exceeds 10 minutes.")
                print(f"       Enabling every-base progress logging and verification crash ledger.")
            else:
                print_interval = 10
            first_base_timed = True

        # Write current state to ledger ONLY on long-running multi-day verification tracks
        if print_interval == 1:
            try:
                with open(recovery_file, "w") as wf:
                    json.dump({"verified_bases_count": actual_global_idx}, wf)
            except Exception:
                pass

        # Terminal Progress Update Checkpoint
        if actual_global_idx % print_interval == 0 or actual_global_idx == k_total:
            elapsed = time.perf_counter() - start_calc_time
            # Adjust average calculations to factor in pre-existing recovered items cleanly
            processed_this_session = current_offset
            avg_time = elapsed / processed_this_session
            remaining_iters = k_total - actual_global_idx
            print(f"   -> Progress: {actual_global_idx}/{k_total} bases verified. [Ave. {avg_time:.3f} sec/base | ETR: {format_etr(remaining_iters * avg_time)}]")

    end_calc_time = time.perf_counter()
    total_calc_time = end_calc_time - start_calc_time
    avg_speed = total_calc_time / len(bases_to_test) if len(bases_to_test) > 0 else 0

    # Upon successful audit completion, remove the temporary ledger file from the disk directory
    if os.path.exists(recovery_file) and verified:
        try:
            os.remove(recovery_file)
        except Exception:
            pass

    final_verdict = "VERIFIED SUCCESSFUL (PROBABLE PRIME)" if verified else "AUDIT FAILED (COMPOSITE OR TAMPERED)"
    fn_expr = expression.replace("**", "^").replace(" ", "")
    out_filename = f"MR_{fn_expr}_verified.txt"

    print("\n" + "="*60)
    print(f" AUDIT VERDICT: {final_verdict}")
    print(f" Verification Type: {audit_type}")
    print("="*60)
    print(f"Core Calculation Time: {total_calc_time:.4f} seconds")
    print(f"Exporting independent verification log to: '{out_filename}'")
    print("="*60)

    try:
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write("================================================================================\n")
            f.write("                       INDEPENDENT VERIFICATION LOG RECORD                      \n")
            f.write("================================================================================\n\n")
            f.write(f"Source Certificate File    : {cert_filename}\n")
            f.write(f"Target Number Expression   : {expression}\n")
            f.write(f"Verification Type Conducted: {audit_type}\n")
            f.write(f"Total Unique Bases Checked : {len(bases)}\n")
            f.write(f"Audit Status / Verdict     : {final_verdict}\n\n")
            f.write("--- AUDIT ENVIRONMENT SIGNATURE (VERIFIER HARDWARE) ---\n")
            f.write(f"Verifier CPU Model         : {verifier_hw['cpu']}\n")
            f.write(f"Verifier RAM Capacity      : {verifier_hw['ram']}\n")
            f.write(f"Verifier Operating System  : {verifier_hw['os']}\n\n")
            f.write("--- PERFORMANCE ANALYSIS ---\n")
            f.write(f"Total Audit Crunch Time    : {total_calc_time:.4f} seconds\n")
            f.write(f"Average Speed Per Base     : {avg_speed:.4f} seconds/base\n\n")
            f.write("================================================================================\n")
            f.write("      This file proves the target was independently audited and verified.     \n")
            f.write("================================================================================\n")
    except Exception as e:
        print(f"Error writing verification log file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        verify_certificate(sys.argv[1])
    else:
        file_input = input("Enter the path to the certification text file:\n> ")
        verify_certificate(file_input.strip())


## The most memory this process has held resident, for the measuring scripts. Godot's own counters
## leave out what the extension allocates, so this asks the operating system: `VmHWM` in
## `/proc/self/status` on Linux, the process's peak working set on Windows.

## The peak resident memory in MB of 10^6 bytes, or -1 when the system did not say.
static func peak_resident_mb() -> float:
	if OS.get_name() == "Windows":
		var output := []
		var command := "(Get-Process -Id %d).PeakWorkingSet64" % OS.get_process_id()
		if OS.execute("powershell", ["-NoProfile", "-Command", command], output) != 0 or output.is_empty():
			return -1.0
		return output[0].strip_edges().to_float() / 1e6
	# A /proc file has no length, so it is read a line at a time to its end.
	var status := FileAccess.open("/proc/self/status", FileAccess.READ)
	if status == null:
		return -1.0
	while not status.eof_reached():
		var line := status.get_line()
		if line.begins_with("VmHWM:"):
			# The kernel's kB are KiB.
			return line.trim_prefix("VmHWM:").strip_edges().trim_suffix("kB").strip_edges().to_float() * 1024.0 / 1e6
	return -1.0

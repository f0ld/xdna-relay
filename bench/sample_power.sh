#!/usr/bin/env bash
# Sample SoC package power (amdgpu PPT) and iGPU sclk every 0.25 s until killed.
# Output: t_s,ppt_w,sclk_mhz
H=$(grep -l amdgpu /sys/class/hwmon/hwmon*/name | head -1 | xargs dirname)
t0=$(date +%s.%N)
echo "t_s,ppt_w,sclk_mhz"
while :; do
  echo "$(echo "$(date +%s.%N) - $t0" | bc),$(( $(cat $H/power1_input) / 1000000 )),$(( $(cat $H/freq1_input) / 1000000 ))"
  sleep 0.25
done

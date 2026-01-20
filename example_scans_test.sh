#!/bin/bash

set -e

mappings=(
    "2024_11_13_21_42_41 Girton_large_study_room"
    "2025_01_20_08_44_07 BoardRoom_CUED"
    "2024_11_13_21_38_09 Girton_Study_Room"
    "2025_03_16_10_29_01 Girton_Common_Room"
    "2025_05_05_08_42_28 Darwin_BedRoom"
    "2026_01_15_15_16_00 SigProc_meeting_room"
    "2024_11_29_13_53_13 Meeting_Room_CUED_Lab"
    "2025_03_14_18_54_12 SigProc_Tea_Room"
    "2025_05_01_18_53_35 SigProc_Tea_Room_Day_Light"
)

for line in "${mappings[@]}"; do
    bash script.sh "scans/$(echo $line | awk '{print $1}')" "$(echo $line | awk '{print $2}')"
done

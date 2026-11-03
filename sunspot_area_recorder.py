#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"Created on Wed Mar 25 17:24:00 2026"

#@author: lgaiger



import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
from datetime import datetime, timedelta

VIDEO_FOLDER = "/Users/lgaiger/Downloads"
VIDEO_PATTERN = "*.mp4"

SKIP_FRAMES = 1
MIN_SPOT_AREA = 10
OUTPUT_FOLDER = "sunspot_results"

# THRESHOLDS:
# "adaptive" is local background ratio threshold to account limb darkening and sphere
# "fixed"   is first attempt global threshold 
THRESHOLD_METHOD = "adaptive"

THRESHOLD = 80                 

SPOT_RATIO = 0.85              # spot if pixel less than 85% of local photosphere
                               
                               #   higher  more noise
BG_SIGMA_FRACTION = 0.08       
LIMB_MARGIN_FRACTION = 0.03
MIN_SOLAR_AREA = 5000          # ignore frames where the disk is tiny 

# 0 = off, overlay img
DEBUG_EVERY = 0

# Updated start date to 2016
START_DATE = datetime(2016, 1, 1)
END_DATE = datetime(2021, 12, 31)  # extend later


def sort_videos_by_name(video_files):
    return sorted(video_files)

#Processing

def find_sun_disk(gray_image):
    _, binary = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0

    sun = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(sun)

    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [sun], -1, 255, -1)

    return mask, area


def _odd(n, minimum=3):
    n = max(minimum, int(round(n)))
    return n if n % 2 else n + 1


def _masked_blur(img, weight, sigma):
   
    num = cv2.GaussianBlur(img * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-6)


def _count_spot_area(binary, min_area):
   
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)

    total_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            total_area += area

    return total_area, binary


def detect_spots_fixed(image, mask, min_area, threshold):
   
    
    masked = cv2.bitwise_and(image, image, mask=mask)

    _, spots = cv2.threshold(masked, threshold, 255, cv2.THRESH_BINARY_INV)
    spots = cv2.bitwise_and(spots, spots, mask=mask)

    return _count_spot_area(spots, min_area)


def detect_spots_adaptive(image, mask, solar_area, min_area,
                          spot_ratio=SPOT_RATIO,
                          bg_sigma_fraction=BG_SIGMA_FRACTION,
                          limb_margin_fraction=LIMB_MARGIN_FRACTION):
    
    
    radius = np.sqrt(solar_area / np.pi)

    # Erode the disk mask to remove the limb edge 
    margin = _odd(2 * radius * limb_margin_fraction)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin, margin))
    inner = cv2.erode(mask, erode_kernel)
    if cv2.countNonZero(inner) == 0:
        inner = mask

    img = image.astype(np.float32)
    weight = (inner > 0).astype(np.float32)
    sigma = max(3.0, radius * bg_sigma_fraction)

    # rough background including everything on the disk
    background = _masked_blur(img, weight, sigma)

    ## exclude dark candidates so big spot groups don't lower their own background
    candidates = (img < spot_ratio * background) & (weight > 0)
    weight_clean = weight * (~candidates).astype(np.float32)
    background = _masked_blur(img, weight_clean, sigma)

    ratio = img / np.maximum(background, 1.0)
    spots = ((ratio < spot_ratio) & (inner > 0)).astype(np.uint8) * 255

    return _count_spot_area(spots, min_area)


def detect_spots(image, mask, solar_area, min_area):
    if THRESHOLD_METHOD == "adaptive":
        return detect_spots_adaptive(image, mask, solar_area, min_area)
    return detect_spots_fixed(image, mask, min_area, THRESHOLD)


def save_debug_overlay(frame, mask, spots_binary, date, percent):
    debug_dir = os.path.join(OUTPUT_FOLDER)
    os.makedirs(debug_dir, exist_ok=True)

    overlay = frame.copy()
    disk_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    spot_contours, _ = cv2.findContours(spots_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, disk_contours, -1, (0, 255, 0), 1)
    cv2.drawContours(overlay, spot_contours, -1, (0, 0, 255), 1)
    cv2.putText(overlay, f"{date:%Y-%m-%d}  {percent:.3f}%  [{THRESHOLD_METHOD}]",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imwrite(os.path.join(debug_dir), overlay)

# vids

def process_all_videos(video_files):

    all_dates = []
    all_percentages = []
    current_day = 0

    print(f"\nProcessing videos for period: 2016 - 2021")

    for video_file in video_files:
        print(f"Processing {os.path.basename(video_file)}")

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Error opening video")
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mask, solar_area = find_sun_disk(gray)
            if mask is None or solar_area < MIN_SOLAR_AREA:
                continue

            spot_area, spots_binary = detect_spots(gray, mask, solar_area, MIN_SPOT_AREA)

            percent = (spot_area / solar_area) * 100

            date = START_DATE + timedelta(days=current_day)

            # Stop if we exceed end date
            if date > END_DATE:
                print(f"Reached end date {END_DATE.strftime('%Y-%m-%d')}")
                cap.release()
                return all_dates, np.array(all_percentages)

            if DEBUG_EVERY and current_day % DEBUG_EVERY == 0:
                save_debug_overlay(frame, mask, spots_binary, date, percent)

            all_dates.append(date)
            all_percentages.append(percent)

            current_day += 1

        cap.release()

    return all_dates, np.array(all_percentages)

###PLOT

def create_nasa_style_plot(dates, percentages):

    # Convert dates to years for x
    x_vals = np.array([d.year + (d.timetuple().tm_yday - 1) / 365.25 for d in dates])

    plt.figure(figsize=(18, 6))

    # KEY: vertical spike lines
    plt.vlines(x_vals, 0, percentages,
               color='black', linewidth=0.3)

    # Axis l
    plt.ylim(0, max(0.5, np.max(percentages)*1.1))

    ##Set x-axis limits to exactly 2016-2021
    plt.xlim(2015.5, 2021.5)

    # X 
    tick_positions = [2016, 2017, 2018, 2019, 2020, 2021]
    tick_labels = ['2016', '2017', '2018', '2019', '2020', '2021']

    plt.xticks(tick_positions, tick_labels, fontsize=11, fontweight='bold')

    # Labels (match style)
    plt.xlabel("YEAR", fontsize=12, fontweight='bold')
    plt.ylabel("Sunspot Area (%)", fontsize=12, fontweight='bold')

    plt.title("AVERAGE DAILY SUNSPOT AREA (% OF VISIBLE HEMISPHERE)\n2016 - 2021",
              fontsize=12, fontweight='bold')

    # 
    plt.grid(axis='y', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.grid(axis='x', linestyle='--', linewidth=0.3, alpha=0.2)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    # statistics box
    stats_text = f"Period: 2016 - 2021\n"
    stats_text += f"Days: {len(percentages):,}\n"
    stats_text += f"Max: {np.max(percentages):.3f}%\n"
    stats_text += f"Mean: {np.mean(percentages):.3f}%\n"
    stats_text += f"Method: {THRESHOLD_METHOD}"

    plt.text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    output_file = os.path.join(OUTPUT_FOLDER, f"sunspot_nasa_style_2016_2021_{THRESHOLD_METHOD}.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.show()

    print(f"Saved: {output_file}")







def save_csv_data(dates, percentages):
    csv_file = os.path.join(OUTPUT_FOLDER, f"sunspot_data_2016_2021_{THRESHOLD_METHOD}.csv")

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Date', 'Year', 'Area %'])

        for d, p in zip(dates, percentages):
            writer.writerow([d.strftime("%Y-%m-%d"), d.year, f"{p:.6f}"])

    print(f"CSV saved: {csv_file}")

# MAIN
def main():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"\n📁 Output folder: {OUTPUT_FOLDER}")
    print(f"📅 Target period: 2016 - 2021")

    video_pattern = os.path.join(VIDEO_FOLDER, VIDEO_PATTERN)
    video_files = glob.glob(video_pattern)

    if not video_files:
        print("No videos found")
        sys.exit(1)

    video_files = sort_videos_by_name(video_files)

    print(f"Found {len(video_files)} videos")

    dates, percentages = process_all_videos(video_files)

    if len(percentages) == 0:
        print("No data extracted")
        return

    save_csv_data(dates, percentages)

    create_nasa_style_plot(dates, percentages)

    print("\n" + "="*50)
    print("SUMMARY - 2016-2021")
    print("="*50)
    print(f"Days: {len(percentages)}")
    print(f"Years: {len(percentages)/365.25:.2f}")
    print(f"Max: {np.max(percentages):.4f}%")
    print(f"Min: {np.min(percentages):.4f}%")
    print(f"Avg: {np.mean(percentages):.4f}%")

    # Yearly statistics
    print(f"\nYearly Averages:")
    for year in range(2016, 2022):
        year_data = [p for p, d in zip(percentages, dates) if d.year == year]
        if year_data:
            print(f"  {year}: Avg={np.mean(year_data):.3f}%, Max={np.max(year_data):.3f}%")


if __name__ == "__main__":
    main()

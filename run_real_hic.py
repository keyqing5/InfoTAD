import numpy as np
import src.detector_v2 as caller
import sys
import argparse


def main_proc(file_name_dir, output_root, start, step_scal=1, resolution=25000, KR=False, chr_name="chr1"):
    input_matrix = np.loadtxt(file_name_dir)
    resolution = int(resolution)
    start_pos = int(start/ resolution)
    file_name = f"{output_root}/{chr_name}_{resolution}kb_{start_pos}_step{step_scal}"
    detect = caller.TADDetector(input_matrix=input_matrix, step_scal=step_scal, KR_norm=KR)
    detect.construct_one_layer(filename=file_name, start_pos=start_pos)
    min_entro = detect.min_entro
    print("Minimum Infomap entropy:", min_entro)

def parse_args():
    parser = argparse.ArgumentParser(description="Process TAD detection parameters.")
    parser.add_argument('-i', '--input', required=True, help='Path to the data file')
    parser.add_argument('-o', '--output', required=True, help='Output root directory')
    parser.add_argument('-s', '--start', type=int, required=True, help='Start site')
    parser.add_argument('-c', '--chrname', default='chr1', help='Chromosome name (default: chr1)')
    parser.add_argument('-r', '--resolution', type=int, default=25000, help='Resolution in base pairs (default: 25000)')
    parser.add_argument('-step', '--step_scal', type=float, default=1, help='Step scale parameter (default: 1)')
    parser.add_argument('-kr', '--KR', action='store_true', help='Use KR normalization (default: False)')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main_proc(
        file_name_dir=args.input,
        output_root=args.output,
        start=args.start,
        chr_name=args.chrname,
        resolution=args.resolution,
        step_scal=args.step_scal,
        KR=args.KR
    )

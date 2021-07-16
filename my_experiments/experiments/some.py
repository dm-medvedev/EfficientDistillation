import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Look for results')
	parser.add_argument('res_pth', type=str, help='pth to res file')
	return parser.parse_args()

if __name__ == "__main__":
	print(parse_args())
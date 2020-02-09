import os
from random import shuffle

def main(args):
  # Saves a text file with the following format per line: [file name] [label]
  # Where [label] is one of 1, 2, or 3 corresponding to train, val, and test respectively.

  files = [ f.name for f in os.scandir(args.image_pair_dir) if f.name.endswith(".png") ]

  file_set = set()

  for f in files:
    file_set.add(f[:f.rindex("_")])

  files = list(file_set)
  shuffle(files)

  num_files = len(files)

  test = int(num_files * args.test_portion)
  val = int(num_files * args.val_portion)

  output = []

  for i, f in enumerate(files):
    label = 1
    if i < test:
      label = 3
    elif i < test + val:
      label = 2
    output.append("{} {:d}".format(f, label))

  f = open(args.output_file, "w")
  f.write("\n".join(output))
  f.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate video subsets')
    parser.add_argument('--image_pair_dir',type=str,default=None,
                        help='directory containing image pairs')
    parser.add_argument('--output_file',type=str,default=None,
                        help='file to write subset information to')
    parser.add_argument('--test_portion',type=float,default=0.1,
                        help='percentage of data to use for test subset')
    parser.add_argument('--val_portion',type=float,default=0.1,
                        help='percentage of data to use for validation subset')
    args = parser.parse_args()
    main(args)

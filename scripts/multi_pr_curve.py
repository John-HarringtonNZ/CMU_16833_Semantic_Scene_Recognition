import argparse
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from eval_proposals import get_score_match_pairs, precision_at_n



if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--yaml_files", nargs='+', required=True
  )
  args = parser.parse_args()

  fig = plt.figure(figsize=(20,12))
  for i, fname in enumerate(args.yaml_files):
    print(f'plotting pr curve for {fname}')
    f = open(fname)
    data = yaml.load(f, Loader=yaml.SafeLoader)
    filtered_proposals = {}
    for target, proposals in data.items():
        filtered_proposals[target] = proposals
    score_match_pairs = get_score_match_pairs(filtered_proposals)

    if 'Noise-free' in fname:
       linestyle='-'
    else:
       linestyle='--'

    if i == 0:
       linestyle=':'

    precision, recall, _ = precision_recall_curve(score_match_pairs[:,0], score_match_pairs[:,1])
    plt.plot(recall, precision, label=fname.split('.')[0].split('/')[-1], linestyle=linestyle)

    print(f'mAP: {average_precision_score(score_match_pairs[:,0], score_match_pairs[:,1])}')

    print('Precision @ N:')
    for n in [1, 3, 5, 10, 20, 50, 100]:
      prec_n, n_targets = precision_at_n(filtered_proposals, n)
      print(f'Precision at {n}: {prec_n} ({n_targets} targets)')

  plt.xlabel('Recall', fontsize=18)
  plt.ylabel('Precision', fontsize=18)
  plt.title('Precision-Recall Curve', fontsize=20)
  plt.xticks(fontsize=16)
  plt.yticks(fontsize=16)
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.legend(fontsize=18)
  fig.savefig('pr_curve.png')

  plt.show()

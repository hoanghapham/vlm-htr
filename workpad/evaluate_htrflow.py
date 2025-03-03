gt_pages_dir = "/Users/hoanghapham/Projects/visual-language-models/data/sample_image_page"
candidate_pages_dir = "/Users/hoanghapham/Projects/visual-language-models/workpad/outputs/page/Göteborgs_poliskammare_före_1900__A_II__1a__1868_"

from htrflow.evaluate import CER, WER, BagOfWords, read_xmls

candidates = read_xmls(candidate_pages_dir)
ground_truth = read_xmls(gt_pages_dir)

page = list(ground_truth.keys())[0]

wer_metric = WER()

value = wer_metric(ground_truth[page], candidates[page])

# %%

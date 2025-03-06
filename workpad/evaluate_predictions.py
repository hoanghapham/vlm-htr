#%%
from pagexml.parser import parse_pagexml_file
from dotenv import dotenv_values
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent

env_values = dotenv_values(PROJECT_DIR / ".env")

data_dir = Path(env_values["POLIS_DATA_DIR"])
img_path = data_dir / "images/Göteborgs_poliskammare_före_1900__A_II__12__1880_/Göteborgs_poliskammare_före_1900__A_II__12__1880__30003111_00003.tif"
xml_path = data_dir / "page_xmls/Göteborgs_poliskammare_före_1900__A_II__12__1880_/Göteborgs_poliskammare_före_1900__A_II__12__1880__30003111_00003.xml"

xml = parse_pagexml_file(xml_path)


# %%
content = ""
for line in xml.get_lines():
    if line.text:
        content += line.text
        content += "\n"

# %%
import re
split_pattern = r"\s"

gt_tokens = [token.lower() for token in re.split(split_pattern, content)]


# %%

chatgpt_output = """
Rapportbok år 1880

Fridgidsdomätaren Alfred Andersson, boende i
huset N:o 120 Litt M. vid St. Bangatan, har den 9:de
December anmält, att den 12:te November
i hans bostad från honom tillgripits
en skjortrock af blått rutigtt bomulltyg värd 30 öre,
ett par byxor och väst af blått randigt ylletyg värda 1 R:dr
en vinterpälsrock af blått kläde, värd 20 R:dr,
samt att stölden föröfvats af Fridgidsdomätarens
Bengt Persson, hvilken bott hos Andersson
i åtta dagar, men afflyttat från bostaden
samma dag kläderna tillgrepos, hvarvid han
qvarlemnat sina egna gamla kläder.

Sedan upplysning erhållits, att Persson begifvit
sig till Lund, afsändes den 12 December skrifvelse
till Stadsfiskalen derstädes med begäran om
Perssons efterspanande och häktande; men då
N. Bofas i Malmö med jernbanetrafik be-
stämdes afsändes Persson i hufvudsak
tillfrågad i Lund, hvilken icke var förvaras
af närvarande Cellfängelse.

Persson, hvilken enligt förut gifna upplysningar,
icke är straffad, men för undergående af
tukt- och arbetsanstalt blifvit förpassad till
Landskrona, har erkänt att han den 12:te sist-
l. November försvunnit i resa till Wenersborg
afsändning såsom af Andersson på sina
skjortrock och öfverrock, jämte 3 R:dr 12 öre,
att han den 12. November för omyndig stånd
att genom tillgripa Andersson hos blått
ylletyg i deras gemensamma bostad stigit
liksom byxor och väst, att han derefter med
bantåg afgått till Wenersborg der han uppe-
hållit sig till den 24. samma månad.
"""

chatgpt_tokens = [token.lower() for token in re.split(split_pattern, chatgpt_output)]


def bow_match_rate(pred: list[str], ground_truth: list[str]):
    matched = 0
    for token in pred:
        if token in ground_truth:
            matched += 1

    print(f"Matched {matched}/{len(ground_truth)}, ratio {matched/len(ground_truth):4f}")

bow_match_rate(chatgpt_tokens, gt_tokens)
# %%


htrflow_xml_path = PROJECT_DIR / "output/page/Göteborgs_poliskammare_före_1900__A_II__12__1880_/Göteborgs_poliskammare_före_1900__A_II__12__1880__30003111_00003.xml"
htrflow_xml = parse_pagexml_file(htrflow_xml_path)

htrflow_tokens = []

for line in htrflow_xml.get_words():
    if line.text:
        htrflow_tokens += [token.lower() for token in re.split(split_pattern, line.text)]

bow_match_rate(htrflow_tokens, gt_tokens)
# %%

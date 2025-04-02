
# Tasks
Florence 2 line-based HTR: <SwedishHTR>
Old prompt: <SwedishHTR>Print out the text in this image

# OD
- PageXML 
    - bbox: XYWH (X, Y, width, height)
    - polygon: counter clockwise, [(x, y)...]
- Florence:
    - bbox: xyxy (TLBR): (xmin, ymin, xmax, ymax)



# Datasets:

bergskollegium_adv:     Riksarkivet/bergskollegium_advokatfiskalskontoret_seg
bergskollegium_rel:     Riksarkivet/bergskollegium_relationer_och_skrivelser_seg
frihetstidens:          Riksarkivet/frihetstidens_utskottshandlingar
gota_hovratt:           Riksarkivet/gota_hovratt_seg
jonkopings_radhusratts: Riksarkivet/jonkopings_radhusratts_och_magistrat_seg
krigshovrattens:        Riksarkivet/krigshovrattens_dombocker_seg
poliskammare:           Riksarkivet/goteborgs_poliskammare_fore_1900
svea_hovratt:           Riksarkivet/svea_hovratt_seg
trolldoms:              Riksarkivet/trolldomskommissionen_seg


- split by sources:

train_val_dirs = [
    "trolldoms",
    "svea_hovratt",
    "bergskollegium_rel",
    "poliskammare",
]

test_dirs = [
    "jonkopings_radhusratts",
    "gota_hovratt",
    "bergskollegium_adv",
    "frihetstidens",
    "krigshovrattens",
]


- More even color distribution
subset	total files	split
gota_hovratt	53	test
bergskollegium_rel__test	750	test
krigshovrattens	346	test
bergskollegium_rel__train_val	750	train-val
poliskammare	5410	train-val
jonkopings_radhusratts	41	train-val
bergskollegium_adv	55	train-val
frihetstidens	245	train-val
trolldoms	770	train-val
svea_hovratt	1247	train-val
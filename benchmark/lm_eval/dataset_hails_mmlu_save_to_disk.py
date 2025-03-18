import datasets

dataset_names = [
	'abstract_algebra',
	'anatomy',
	'astronomy',
	'business_ethics',
	'clinical_knowledge',
	'college_biology',
	'college_chemistry',
	'college_computer_science',
	'college_mathematics',
	'college_medicine',
	'college_physics',
	'computer_security',
	'conceptual_physics',
	'econometrics',
	'electrical_engineering',
	'elementary_mathematics',
	'formal_logic',
	'global_facts',
	'high_school_biology',
	'high_school_chemistry',
	'high_school_computer_science',
	'high_school_european_history',
	'high_school_geography',
	'high_school_government_and_politics',
	'high_school_macroeconomics',
	'high_school_mathematics',
	'high_school_microeconomics',
	'high_school_physics',
	'high_school_psychology',
	'high_school_statistics',
	'high_school_us_history',
	'high_school_world_history',
	'human_aging',
	'human_sexuality',
	'international_law',
	'jurisprudence',
	'logical_fallacies',
	'machine_learning',
	'management',
	'marketing',
	'medical_genetics',
	'miscellaneous',
	'moral_disputes',
	'moral_scenarios',
	'nutrition',
	'philosophy',
	'prehistory',
	'professional_accounting',
	'professional_law',
	'professional_medicine',
	'professional_psychology',
	'public_relations',
	'security_studies',
	'sociology',
	'us_foreign_policy',
	'virology',
	'world_religions'
]

dataset_path = "hails/mmlu_no_train"
save_disk_path = "hails_mmlu_no_train"

for i in range(len(dataset_names)):
	name = dataset_names[i]
	dataset = datasets.load_dataset(
		path=dataset_path,
		name=name,
		trust_remote_code=True,)

	dataset.save_to_disk(f"{save_disk_path}/{name}")
	print(f"dataset: {dataset}, name: {name}")

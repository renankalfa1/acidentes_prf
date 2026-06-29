import sweetviz as sv
import pandas as pd

acidentes = pd.read_parquet(r'data\processed\acidentes.parquet')

my_report = sv.analyze(acidentes)
my_report.show_html('data/eda_acidentes.html')

import duckdb
import os
import time

# Configurações de caminhos
arquivos_acidentes = 'data/raw/*.csv'
saida_arquivo = 'data/processed'

def converter_csv_para_parquet(pasta_arquivos=str(), nome_arquivo='temp'):
    print("Iniciando a conversão...")
    
    # Cria conexão com o Duckdb
    con = duckdb.connect()

    # Lê todos os CSVs da pasta e salva em um único arquivo Parquet
    try:
        con.execute(f"""
            COPY (
                SELECT * FROM read_csv_auto('{pasta_arquivos}', 
                                   encoding='ISO_8859_2',
                                   union_by_name=true)
            ) 
            TO '{saida_arquivo}/{nome_arquivo}.parquet' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');
        """)
        
        tamanho_final = os.path.getsize(f'{nome_arquivo}.parquet') / (1024**3) # Tamanho em GB
        
        print(f"Sucesso!")
        print(f"Arquivo gerado: {nome_arquivo} ({tamanho_final:.2f} GB)")

    except Exception as e:
        print(f"Erro na conversão: {e}")

if __name__ == "__main__":
    converter_csv_para_parquet(pasta_arquivos=arquivos_acidentes, nome_arquivo='acidentes')

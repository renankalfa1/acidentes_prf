import duckdb
import os

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
                SELECT
                    TRY_CAST(id AS INTEGER)                                         AS id,
                    TRY_CAST(data_inversa AS DATE)                                  AS data_inversa,
                    TRY_CAST(tipo_acidente AS VARCHAR)                              AS tipo_acidente,
                    TRY_CAST(dia_semana AS VARCHAR)                                 AS dia_semana,
                    TRY_CAST(fase_dia AS VARCHAR)                                   AS fase_dia,
                    TRY_CAST(causa_acidente AS VARCHAR)                             AS causa_acidente,
                    TRY_CAST(REPLACE(REPLACE(latitude,  ',', '.'), ' ', '') AS DOUBLE) AS latitude,
                    TRY_CAST(REPLACE(REPLACE(longitude, ',', '.'), ' ', '') AS DOUBLE) AS longitude
                FROM read_csv_auto(
                    '{pasta_arquivos}',
                    encoding='ISO_8859_2',
                    union_by_name=true
                )
            )
            TO '{saida_arquivo}/{nome_arquivo}.parquet' (FORMAT 'PARQUET', COMPRESSION 'ZSTD');
        """)

        tamanho_final = os.path.getsize(f'{saida_arquivo}/{nome_arquivo}.parquet') / (1024**3)

        print(f"Sucesso!")
        print(f"Arquivo gerado: {saida_arquivo}/{nome_arquivo}.parquet ({tamanho_final:.2f} GB)")

    except Exception as e:
        print(f"Erro na conversão: {e}")

if __name__ == "__main__":
    converter_csv_para_parquet(pasta_arquivos=arquivos_acidentes, nome_arquivo='acidentes')

import pandas as pd

def process_covid_data(input_path='owid-covid-data.csv', output_path='germany_covid_processed.csv'):
    """
    加载OWID新冠数据，筛选、清洗并计算德国在特定时间段内的疫情数据，
    最终生成用于SEIAR模型拟合的干净数据表。
    """
    print(f"开始处理数据，从 '{input_path}' 读取...")

    try:
        # --- 1. 加载与筛选 ---
        df = pd.read_csv(input_path)

        # 将date列转换为datetime对象，便于筛选
        df['date'] = pd.to_datetime(df['date'])

        # a. 筛选 location 为 'Germany'
        df_germany = df[df['location'] == 'Germany'].copy()

        # b. 筛选 date 在 '2020-03-01' 和 '2021-03-01' 之间
        start_date = '2020-03-01'
        end_date = '2021-03-01'
        mask = (df_germany['date'] >= start_date) & (df_germany['date'] <= end_date)
        df_processed = df_germany.loc[mask].copy()
        
        print(f"成功筛选出德国从 {start_date} 到 {end_date} 的数据。")

        # --- 2. 数据清洗与平滑 ---

        # a. 处理缺失值：将 new_cases 列的 NaN 填充为 0
        df_processed['new_cases'] = df_processed['new_cases'].fillna(0)

        # b. 计算7日移动平均，生成平滑后的新增病例数
        df_processed['new_cases_smoothed'] = df_processed['new_cases'].rolling(window=7, center=True).mean()
        # 移动平均会在数据序列的开头和结尾产生NaN，我们也需要填充它们
        # 使用前后填充（bfill/ffill）来处理这些边界情况
        df_processed['new_cases_smoothed'] = df_processed['new_cases_smoothed'].fillna(method='bfill').fillna(method='ffill')
        
        print("已完成新增病例的平滑处理（7日移动平均）。")

        # --- 3. 计算核心拟合数据 ---

        # 计算估算的现存感染人数（19天滚动求和）
        infection_period = 19
        df_processed['infectious_pool'] = df_processed['new_cases_smoothed'].rolling(window=infection_period).sum()
        # 同样，处理滚动求和在开头产生的NaN
        df_processed['infectious_pool'] = df_processed['infectious_pool'].fillna(0)

        print(f"已完成现存感染人数的估算（{infection_period}日滚动求和）。")

        # --- 4. 整理并保存 ---
        
        # a. 创建从0开始的时间序列列
        df_processed.reset_index(drop=True, inplace=True)
        df_processed['time'] = df_processed.index

        # b. 只保留需要的列
        # 我们还需要'population'列
        final_columns = ['date', 'time', 'new_cases_smoothed', 'infectious_pool', 'population']
        df_final = df_processed[final_columns].copy()

        # 将date列转回字符串格式，以便保存
        df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')

        # c. 保存到新的CSV文件
        df_final.to_csv(output_path, index=False)
        
        print(f"\n处理完成！干净的数据已保存到 '{output_path}'。")
        print("最终数据包含以下列:")
        print(df_final.info())
        print("\n数据预览:")
        print(df_final.head())
        
        return df_final

    except FileNotFoundError:
        print(f"错误：输入文件 '{input_path}' 未找到。请确保文件存在。")
        return None
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        return None

if __name__ == '__main__':
    process_covid_data() 
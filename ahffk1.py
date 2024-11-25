import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class EelHealthDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class EelHealthLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(EelHealthLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class EelHealthPredictor:
    def __init__(self):
        self.sequence_length = 24
        self.scalers = {
            'japonica': {
                'sensor': MinMaxScaler(),
                'quality': MinMaxScaler(),
                'disease': MinMaxScaler()
            },
            'marmorata': {
                'sensor': MinMaxScaler(),
                'quality': MinMaxScaler(),
                'disease': MinMaxScaler()
            }
        }
        self.models = {
            'japonica': None,
            'marmorata': None
        }
    
    def load_and_preprocess_data(self, data_paths):
        """데이터 로드 및 전처리"""
        def load_single_species(path):
            try:
                print(f"\nLoading data from {path}")
                sensor_data = pd.read_csv(f"{path}/sensor_val_tb.csv")
                quality_data = pd.read_csv(f"{path}/water_quality_tb.csv")
                disease_data = pd.read_csv(f"{path}/disease_test_tb.csv")
                
                print("\nPreprocessing data...")
                # 센서 데이터 전처리
                sensor_cols = ['do_mg', 'do_temp', 'ph', 'orp', 'co2_mg', 'air_oxy', 'light_ma']
                print("Processing sensor data...")
                sensor_data['mea_dt'] = pd.to_datetime(sensor_data['mea_dt'])
                sensor_data = sensor_data[['mea_dt', 'tank_id'] + sensor_cols].copy()
                for col in sensor_cols:
                    sensor_data[col] = pd.to_numeric(sensor_data[col], errors='coerce')
                
                # 수질 데이터 전처리
                quality_cols = ['ammo_nitro', 'a_nitric_nitro', 'nitric_nitro',
                            'alkali', 'ss', 'total_bacterial', 'tubidity']
                print("Processing quality data...")
                
                # measure_dt 데이터 검증 및 정리
                print("\nValidating quality data dates...")
                quality_data['measure_dt'] = quality_data['measure_dt'].astype(str)
                # 올바른 날짜 형식만 유지 (YYYYMMDD, 8자리)
                valid_date_mask = (quality_data['measure_dt'].str.len() == 8) & \
                                (quality_data['measure_dt'].str.match(r'^\d{8}$')) & \
                                (quality_data['measure_dt'].str.slice(4,6).astype(int) <= 12) & \
                                (quality_data['measure_dt'].str.slice(6,8).astype(int) <= 31)
                
                if not valid_date_mask.all():
                    invalid_dates = quality_data[~valid_date_mask]['measure_dt']
                    print(f"Removing {len(invalid_dates)} invalid dates: {invalid_dates.unique()}")
                    quality_data = quality_data[valid_date_mask]
                
                # 날짜 변환
                quality_data['datetime'] = pd.to_datetime(quality_data['measure_dt'], format='%Y%m%d')
                quality_data['datetime'] = quality_data.apply(
                    lambda x: x['datetime'].replace(hour=9 if x['day_type'] == 1 else 15), 
                    axis=1
                )
                
                # 질병 데이터 전처리
                disease_cols = ['blood_alt', 'blood_ast', 'blood_glu', 'blood_na',
                            'blood_k', 'blood_ci', 'blood_hb', 'blood_alb', 'blood_ht']
                print("\nProcessing disease data...")
                disease_data['test_dt'] = pd.to_datetime(disease_data['test_dt'].astype(str))
                
                print("\nData shapes after initial preprocessing:")
                print(f"Sensor data: {sensor_data.shape}")
                print(f"Quality data: {quality_data.shape}")
                print(f"Disease data: {disease_data.shape}")
                
                print("\nProcessing by tank...")
                processed_data = []
                
                for tank_id in sensor_data['tank_id'].unique():
                    print(f"\nProcessing tank {tank_id}...")
                    try:
                        # 데이터 필터링
                        tank_sensor = sensor_data[sensor_data['tank_id'] == tank_id].copy()
                        tank_quality = quality_data[quality_data['tank_id'] == tank_id].copy()
                        tank_disease = disease_data[disease_data['tank_id'] == tank_id].copy()
                        
                        print(f"Initial data sizes - Sensor: {len(tank_sensor)}, Quality: {len(tank_quality)}, Disease: {len(tank_disease)}")
                        
                        # 센서 데이터 처리
                        tank_sensor = tank_sensor.set_index('mea_dt')[sensor_cols]
                        tank_sensor = tank_sensor.groupby(level=0).mean()
                        
                        # 수질 데이터 처리
                        tank_quality = tank_quality.set_index('datetime')[quality_cols]
                        tank_quality = tank_quality.groupby(level=0).mean()
                        
                        # 질병 데이터 처리
                        tank_disease = tank_disease.set_index('test_dt')[disease_cols]
                        tank_disease = tank_disease.groupby(level=0).mean()
                        
                        # 결측치 처리
                        tank_sensor = tank_sensor.fillna(method='ffill').fillna(method='bfill')
                        tank_quality = tank_quality.fillna(method='ffill').fillna(method='bfill')
                        tank_disease = tank_disease.fillna(method='ffill').fillna(method='bfill')
                        
                        # 시간 범위 결정
                        start_time = max(tank_sensor.index.min(),
                                    tank_quality.index.min(),
                                    tank_disease.index.min())
                        end_time = min(tank_sensor.index.max(),
                                    tank_quality.index.max(),
                                    tank_disease.index.max())
                        
                        print(f"Time range: {start_time} to {end_time}")
                        
                        # 시간 범위로 데이터 자르기
                        tank_sensor = tank_sensor[start_time:end_time]
                        tank_quality = tank_quality[start_time:end_time]
                        tank_disease = tank_disease[start_time:end_time]
                        
                        # 시간 간격 통일
                        freq = 'H'
                        tank_sensor = tank_sensor.resample(freq).mean()
                        tank_quality = tank_quality.resample(freq).ffill()
                        tank_disease = tank_disease.resample(freq).ffill()
                        
                        # 모든 데이터가 있는 시간대만 선택
                        common_index = tank_sensor.index.intersection(tank_quality.index)
                        common_index = common_index.intersection(tank_disease.index)
                        
                        if len(common_index) > 0:
                            tank_sensor = tank_sensor.loc[common_index]
                            tank_quality = tank_quality.loc[common_index]
                            tank_disease = tank_disease.loc[common_index]
                            
                            # 결측치가 없는 행만 선택
                            valid_rows = (tank_sensor.notna().all(axis=1) & 
                                        tank_quality.notna().all(axis=1) & 
                                        tank_disease.notna().all(axis=1))
                            
                            tank_sensor = tank_sensor[valid_rows]
                            tank_quality = tank_quality[valid_rows]
                            tank_disease = tank_disease[valid_rows]
                            
                            if len(tank_sensor) > 0:
                                processed_data.append({
                                    'tank_id': tank_id,
                                    'sensor': tank_sensor,
                                    'quality': tank_quality,
                                    'disease': tank_disease
                                })
                                print(f"Successfully processed tank {tank_id} with {len(tank_sensor)} samples")
                            else:
                                print(f"No valid samples found for tank {tank_id}")
                        else:
                            print(f"No common timestamps found for tank {tank_id}")
                    
                    except Exception as e:
                        print(f"Error processing tank {tank_id}: {str(e)}")
                        print("Data shapes at error:")
                        print(f"Sensor: {tank_sensor.shape if 'tank_sensor' in locals() else 'Not created'}")
                        print(f"Quality: {tank_quality.shape if 'tank_quality' in locals() else 'Not created'}")
                        print(f"Disease: {tank_disease.shape if 'tank_disease' in locals() else 'Not created'}")
                        continue
                
                if not processed_data:
                    raise ValueError("No tanks were successfully processed")
                
                return processed_data
                
            except Exception as e:
                print(f"\nError in load_single_species:")
                print(f"Error details: {str(e)}")
                raise
        
        return {
            'japonica': load_single_species(data_paths['japonica']),
            'marmorata': load_single_species(data_paths['marmorata'])
        }
    
    def create_sequences(self, processed_data, sequence_length=24):
        """시계열 시퀀스 생성"""
        sequences = []
        targets = []
        
        print("\nCreating sequences...")
        for tank_data in processed_data:
            try:
                # 입력 특성 결합 - tank_id는 이미 제거되어 있으므로 수정
                features = pd.concat(
                    [tank_data['sensor'], tank_data['quality']], 
                    axis=1
                )
                
                targets_data = tank_data['disease']
                
                print(f"\nTank {tank_data['tank_id']}:")
                print(f"Features shape: {features.shape}")
                print(f"Targets shape: {targets_data.shape}")
                
                # 시퀀스 생성
                for i in range(len(features) - sequence_length):
                    seq = features.iloc[i:i+sequence_length].values
                    target = targets_data.iloc[i+sequence_length].values
                    
                    # NaN 체크
                    if not (np.isnan(seq).any() or np.isnan(target).any()):
                        sequences.append(seq)
                        targets.append(target)
                
                print(f"Generated {len(sequences)} sequences")
                
            except Exception as e:
                print(f"Error processing tank {tank_data['tank_id']}: {str(e)}")
                print("Data samples:")
                print("\nFeatures head:")
                print(features.head() if 'features' in locals() else "Not created")
                print("\nTargets head:")
                print(targets_data.head() if 'targets_data' in locals() else "Not created")
                continue
        
        if not sequences:
            raise ValueError("No valid sequences were generated")
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"\nFinal sequences shape: {sequences.shape}")
        print(f"Final targets shape: {targets.shape}")
        
        return sequences, targets
    
    def train_model(self, sequences, targets, species, epochs=100, batch_size=32):
        """모델 학습"""
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, targets, test_size=0.2, random_state=42
        )
        
        # 데이터셋 생성
        train_dataset = EelHealthDataset(X_train, y_train)
        test_dataset = EelHealthDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size
        )
        
        # 모델 초기화
        input_size = sequences.shape[2]  # 특성 수
        hidden_size = 64
        num_layers = 2
        output_size = targets.shape[1]  # 예측할 질병 지표 수
        
        model = EelHealthLSTM(input_size, hidden_size, num_layers, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # 학습
        best_loss = float('inf')
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 검증
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
            
            if epoch % 10 == 0:
                print(f'{species} - Epoch [{epoch}/{epochs}], '
                      f'Train Loss: {train_loss/len(train_loader):.4f}, '
                      f'Val Loss: {val_loss/len(test_loader):.4f}')
            
            # best 모델 저장
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model.state_dict()
        
        # best 모델 복원
        model.load_state_dict(best_model)
        self.models[species] = model
        
        return model
    
    def train_both_species(self, data_paths, epochs=100, batch_size=32):
        """양쪽 종 모두 학습"""
        try:
            # 데이터 로드 및 전처리
            processed_data = self.load_and_preprocess_data(data_paths)
            
            # 자포니카 학습
            print("\nTraining Japonica model...")
            japonica_sequences, japonica_targets = self.create_sequences(
                processed_data['japonica']
            )
            
            print("\nJaponica data shapes:")
            print(f"Sequences: {japonica_sequences.shape}")
            print(f"Targets: {japonica_targets.shape}")
            
            self.train_model(
                japonica_sequences, 
                japonica_targets, 
                'japonica', 
                epochs, 
                batch_size
            )
            
            # 말모라타 학습
            print("\nTraining Marmorata model...")
            marmorata_sequences, marmorata_targets = self.create_sequences(
                processed_data['marmorata']
            )
            
            print("\nMarmorata data shapes:")
            print(f"Sequences: {marmorata_sequences.shape}")
            print(f"Targets: {marmorata_targets.shape}")
            
            self.train_model(
                marmorata_sequences, 
                marmorata_targets, 
                'marmorata', 
                epochs, 
                batch_size
            )
            
            return self.models
        
        except Exception as e:
            print(f"\nError in train_both_species:")
            print(f"Error details: {str(e)}")
            if 'processed_data' in locals():
                print("\nProcessed data summary:")
                for species in processed_data:
                    print(f"\n{species.capitalize()}:")
                    print(f"Number of tanks: {len(processed_data[species])}")
                    for tank_data in processed_data[species]:
                        print(f"\nTank {tank_data['tank_id']}:")
                        print(f"Sensor shape: {tank_data['sensor'].shape}")
                        print(f"Quality shape: {tank_data['quality'].shape}")
                        print(f"Disease shape: {tank_data['disease'].shape}")
            raise
    
def main():
    # 데이터 경로 설정
    data_paths = {
        'japonica': "D:/project/bam/japonica",
        'marmorata': "D:/project/bam/marmorata"
    }
    
    # 예측기 초기화 및 학습
    predictor = EelHealthPredictor()
    
    try:
        # 학습 실행
        print("\nStarting training process...")
        models = predictor.train_both_species(data_paths, epochs=100, batch_size=32)
        print("\nTraining completed successfully!")
        
        return predictor, models
    
    except Exception as e:
        print(f"\nError occurred during training:")
        print(f"Error details: {str(e)}")
        return None, None

if __name__ == "__main__":
    predictor, models = main()
    
    if predictor and models:
        print("\nModel training completed successfully!")
        print("Available models:")
        for species, model in models.items():
            if model is not None:
                print(f"- {species.capitalize()} model is ready")
            else:
                print(f"- {species.capitalize()} model failed to train")
    else:
        print("\nTraining failed. Please check the error messages above.")
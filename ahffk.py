import os
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

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
        
        # 양방향 LSTM 사용
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 더 복잡한 FC 레이어
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class EelHealthPredictor:
    def __init__(self):
        # 현재 시간을 포함한 결과 디렉토리 이름 생성
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = f"results_{current_time}"
        
        # 결과 디렉토리 생성
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            print(f"Created results directory: {self.results_dir}")
        except Exception as e:
            print(f"Error creating results directory: {str(e)}")
            self.results_dir = "results_default"
            os.makedirs(self.results_dir, exist_ok=True)
        
        # 시퀀스 길이 설정
        self.sequence_length = 24
        
        # 스케일러 초기화
        self.scalers = {
            'japonica': {
                'features': StandardScaler(),
                'targets': StandardScaler()
            },
            'marmorata': {
                'features': StandardScaler(),
                'targets': StandardScaler()
            }
        }
        
        # 모델 초기화
        self.models = {
            'japonica': None,
            'marmorata': None
        }
        
        # 학습 결과 저장을 위한 로그 초기화
        self.training_history = {
            'japonica': {'train_loss': [], 'val_loss': []},
            'marmorata': {'train_loss': [], 'val_loss': []}
        }
        
        print(f"\nInitialized EelHealthPredictor")
        print(f"Results will be saved in: {self.results_dir}")
        
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
                # 입력 특성 결합
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

        # 결과 저장을 위한 디렉토리 생성
        self.results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def scale_data(self, features, targets, species, fit=True):
        """데이터 스케일링"""
        if fit:
            features_scaled = self.scalers[species]['features'].fit_transform(features.reshape(-1, features.shape[-1]))
            targets_scaled = self.scalers[species]['targets'].fit_transform(targets)
        else:
            features_scaled = self.scalers[species]['features'].transform(features.reshape(-1, features.shape[-1]))
            targets_scaled = self.scalers[species]['targets'].transform(targets)
        
        return features_scaled.reshape(features.shape), targets_scaled

    def train_model(self, sequences, targets, species, epochs=100, batch_size=32):
        """모델 학습"""
        # 데이터 스케일링
        sequences_scaled, targets_scaled = self.scale_data(sequences, targets, species)
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            sequences_scaled, targets_scaled, test_size=0.2, random_state=42
        )
        
        # 데이터셋 생성
        train_dataset = EelHealthDataset(X_train, y_train)
        test_dataset = EelHealthDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 모델 초기화
        input_size = sequences.shape[2]
        hidden_size = 128  # 증가된 hidden_size
        num_layers = 3    # 증가된 layer 수
        output_size = targets.shape[1]
        
        model = EelHealthLSTM(input_size, hidden_size, num_layers, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # 학습 기록
        train_losses = []
        val_losses = []
        best_loss = float('inf')
        best_model = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # 학습
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
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
                    predictions.extend(outputs.numpy())
                    actuals.extend(batch_y.numpy())
            
            # 손실 계산
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            # Learning rate 조정
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                print(f'{species} - Epoch [{epoch}/{epochs}], '
                    f'Train Loss: {avg_train_loss:.4f}, '
                    f'Val Loss: {avg_val_loss:.4f}')
            
            # Early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_model = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # 학습 과정 시각화
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'{species} Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{self.results_dir}/training_history_{species}.png')
        plt.close()
        
        # 예측 vs 실제 비교
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # 스케일 복원
        predictions = self.scalers[species]['targets'].inverse_transform(predictions)
        actuals = self.scalers[species]['targets'].inverse_transform(actuals)
        
        # 평가 지표를 저장할 딕셔너리
        metrics_dict = {
            'features': [],
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': []
        }
        
        # 각 출력 변수별 산점도와 평가 지표
        for i in range(predictions.shape[1]):
            # 산점도 생성
            plt.figure(figsize=(8, 8))
            plt.scatter(actuals[:, i], predictions[:, i], alpha=0.5)
            plt.plot([actuals[:, i].min(), actuals[:, i].max()],
                    [actuals[:, i].min(), actuals[:, i].max()],
                    'r--', lw=2)
            plt.title(f'{species} - Feature {i+1} Predictions vs Actuals')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.savefig(f'{self.results_dir}/scatter_{species}_feature_{i+1}.png')
            plt.close()
            
            # 평가 지표 계산
            mse = mean_squared_error(actuals[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actuals[:, i], predictions[:, i])
            r2 = r2_score(actuals[:, i], predictions[:, i])
            
            # 딕셔너리에 추가
            metrics_dict['features'].append(f'Feature_{i+1}')
            metrics_dict['mse'].append(mse)
            metrics_dict['rmse'].append(rmse)
            metrics_dict['mae'].append(mae)
            metrics_dict['r2'].append(r2)
            
            print(f"\n{species} - Feature {i+1} Metrics:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R2 Score: {r2:.4f}")
        
        # 평가 지표를 DataFrame으로 변환하고 저장
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_file = f'{self.results_dir}/{species}_metrics.csv'
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\nMetrics saved to: {metrics_file}")
        
        # 학습 히스토리 저장
        history_dict = {
            'epoch': list(range(len(train_losses))),
            'train_loss': train_losses,
            'val_loss': val_losses
        }
        history_df = pd.DataFrame(history_dict)
        history_file = f'{self.results_dir}/{species}_training_history.csv'
        history_df.to_csv(history_file, index=False)
        print(f"Training history saved to: {history_file}")
        
        # 모델 성능 요약 저장
        summary_file = f'{self.results_dir}/{species}_summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"{species} Model Performance Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training Episodes: {len(train_losses)}\n")
            f.write(f"Final Train Loss: {train_losses[-1]:.6f}\n")
            f.write(f"Final Validation Loss: {val_losses[-1]:.6f}\n\n")
            
            f.write("Feature-wise Metrics:\n")
            f.write("-" * 30 + "\n")
            for i in range(len(metrics_dict['features'])):
                f.write(f"\nFeature {i+1}:\n")
                f.write(f"MSE: {metrics_dict['mse'][i]:.6f}\n")
                f.write(f"RMSE: {metrics_dict['rmse'][i]:.6f}\n")
                f.write(f"MAE: {metrics_dict['mae'][i]:.6f}\n")
                f.write(f"R² Score: {metrics_dict['r2'][i]:.6f}\n")
        
        print(f"Performance summary saved to: {summary_file}")
        
        # best 모델 복원 및 저장
        model.load_state_dict(best_model)
        self.models[species] = model
        torch.save({
            'model_state_dict': best_model,
            'scalers': self.scalers[species]
        }, f'{self.results_dir}/{species}_model.pth')
        
        return model

    # load_and_preprocess_data와 create_sequences 함수는 이전과 동일

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
    data_paths = {
        'japonica': "D:/project/bam/japonica",
        'marmorata': "D:/project/bam/marmorata"
    }
    
    predictor = EelHealthPredictor()
    
    try:
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
        print(f"Results saved in: {predictor.results_dir}")
        for species, model in models.items():
            if model is not None:
                print(f"- {species.capitalize()} model is ready")
            else:
                print(f"- {species.capitalize()} model failed to train")
    else:
        print("\nTraining failed. Please check the error messages above.")
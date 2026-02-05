import yaml
import pickle
from pathlib import Path

from oat.data_module.cosmo_peakweather_dataset import CosmoPeakWeatherConfig, CosmoPeakWeatherDataset

def main():
    
    config_path = "/scratch/mch/cdonner/observation_assimilation_transformer/configs/data/cosmo_peakweather_data.yaml"
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    config = CosmoPeakWeatherConfig(**config_dict)

    dataset = CosmoPeakWeatherDataset(config)
    
    input_transform, output_transform = dataset.get_transform(num_data_points=10)
    
    ROOT = Path(__file__).resolve().parents[1]

    out_file = ROOT / "test" / "oat_test" / f"input_transform.pickle"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "wb") as f:
        pickle.dump(input_transform, f)

    print(f"Saved input_transform object to {out_file}")
    
    out_file = ROOT / "test" / "oat_test" / f"output_transform.pickle"
    
    with open(out_file, "wb") as f:
        pickle.dump(output_transform, f)

    print(f"Saved output_transform object to {out_file}")


if __name__ == "__main__":
    """
    Usage: python -m scripts.create_transform_fixture
    """
    main()
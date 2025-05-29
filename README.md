# Smart Telemetry and Diagnostics System

A privacy-aware, machine-learning-powered edge diagnostics framework that monitors system telemetry and automatically responds to anomalies using human-readable rules and cloud-based advice.

## 🔧 Features

- **Edge-based Telemetry Collection** via secure FastAPI
- **Random Forest Classifier** trained on temperature-driven telemetry data
- **Rule-based Diagnostic Engine** with JSON-defined conditions
- **Cloud Escalation** for critical issues and remote advice using LLM
- **Configurable Privacy Modes**:
  - `disable_telemetry` – disables all processing
  - `local_only` – runs locally without cloud access
  - `share_with_dell` – enables full cloud and edge interaction
- **Mocked Dataset Generation** using Hugging Face for simulation

---

## 📁 Project Structure
```text
├── cloud/                                      # Folder containing cloud related files
│   ├── cloud_rules.json                        # Rule set for cloud-side escalations
│   ├── edge_rules.json                         # Rule set for edge-side 
│   ├── log_err.json                            # Output for unmatched rules in cloud processing
│   ├── random_forest_model.joblib              # Trained model stored for cloud use
│
├── edge/                                       # Folder containing cloud related files
│   ├── edge_rules.json                         # Rule set for local edge-level responses. Copied from cloud.
│   ├── random_forest_model.joblib              # Trained model for edge inference. Copied from cloud.
│
├── doc/                                        # Documentation or design diagrams
│   ├── Architecture_diagram.drawio             # Editable architecture diagram (source file)
│   ├── Architecture_diagram.png                # Exported image version for slides/docs
│   ├── Architecture.docx                       # Supporting architecture document
│   ├── Intelligent_Assistant_Proposal.pptx     # Proposal presentation slide deck
├── src/
│   ├── cloud_service.py                        # Cloud-side logic (e.g., model/rule syncing, escalation)
│   ├── cls_env.py                              # Environment variable loader
│   ├── cls_LLM.py                              # LLM-based rule logic or placeholder. 
│   ├── cls_telemetric.py                       # Rule definitions and telemetry structures
│   ├── eda_transform_train_export.py           # Model training pipeline: preprocessing, feature eng, model export
│   ├── machine_service.py                      # API interaction layer with external machine
│   ├── telemetry_collector.py                  # Edge-side logic: data collection, inference, and rule matching
│   ├── util.py                                 # Shared utility functions
│
├── .env                                        # Environment variables for runtime configuration
├── .env_sample                                 # Template .env for distribution
├── .gitignore                                  # Ignore rules for Git version control
│
├── cert.pem                                    # TLS certificate for secure API communication
├── key.pem                                     # Private key for TLS
│
├── cloud_config.yml                            # Cloud-related settings (API, certs, endpoints)
├── eda_config.yml                              # Config for training pipeline (features, splits)
├── machine_config.yml                          # Input telemetry definitions and testing scenarios
├── telemetry_config.yml                        # Runtime config for telemetry behavior and privacy
│
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project overview and usage guide
```

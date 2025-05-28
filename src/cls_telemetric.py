from pydantic import BaseModel

class cls_SystemDiagnostics(BaseModel):
    cpu_usage: float
    memory_usage: float
    network_traffic: float
    power_consumption: float
    num_executed_instructions: float
    execution_time: float
    energy_efficiency: float
    task_priority: int
    temperature: float
    task_type_compute: bool
    task_type_io: bool
    task_type_network: bool
#!/usr/bin/env python3
"""
Test script for algorithmic innovations
Verifies all algorithms work correctly before running the Streamlit demo
"""

import numpy as np
from agents.advanced_algorithms import (
    DynamicProgrammingOptimizer, 
    SwarmIntelligenceOptimizer, 
    QuantumInspiredOptimizer,
    AlgorithmBenchmark
)

def test_algorithms():
    """Test all algorithms with sample data"""
    print("🧮 Testing Algorithmic Innovations...")
    print("=" * 50)
    
    # Test parameters
    servers = 20
    workloads = np.random.random(servers) * 100
    energy_costs = np.random.random(servers) * 10
    
    print(f"📊 Test Parameters:")
    print(f"   Servers: {servers}")
    print(f"   Workloads: {workloads[:5]}... (showing first 5)")
    print(f"   Energy Costs: {energy_costs[:5]}... (showing first 5)")
    print()
    
    # Test individual algorithms
    algorithms = {
        'Dynamic Programming': DynamicProgrammingOptimizer(),
        'Particle Swarm': SwarmIntelligenceOptimizer(num_particles=10, max_iterations=20),
        'Quantum-Inspired': QuantumInspiredOptimizer(num_qubits=10, max_iterations=20)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"🔄 Testing {name}...")
        try:
            result = algorithm.optimize(servers, workloads, energy_costs)
            results[name] = result
            
                         print(f"   ✅ {name} completed successfully!")
             print(f"   📈 Energy Consumption: {result.energy_consumption:.2f} kWh")
             print(f"   🎯 Performance Score: {result.performance_score:.2f} ops/sec")
             print(f"   ⏱️  Algorithm Used: {result.algorithm_used}")
            print()
            
        except Exception as e:
            print(f"   ❌ {name} failed: {str(e)}")
            print()
    
    # Test benchmark
    print("🏁 Testing Algorithm Benchmark...")
    try:
        benchmark = AlgorithmBenchmark()
        benchmark_results = benchmark.benchmark(servers, workloads, energy_costs)
        
        print("   ✅ Benchmark completed successfully!")
                 for name, data in benchmark_results.items():
             if 'error' not in data:
                 print(f"   📊 {name}: Energy={data['energy_consumption']:.2f} kWh, Performance={data['performance_score']:.2f} ops/sec")
        print()
        
    except Exception as e:
        print(f"   ❌ Benchmark failed: {str(e)}")
        print()
    
    # Summary
    print("📋 Test Summary:")
    print("=" * 50)
    successful = len([r for r in results.values() if r is not None])
    total = len(algorithms)
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful == total:
        print("🎉 All algorithms working correctly! Ready for Streamlit demo.")
        return True
    else:
        print("⚠️  Some algorithms failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = test_algorithms()
    if success:
        print("\n🚀 You can now run: streamlit run streamlit_algorithmic_demo.py")
    else:
        print("\n🔧 Please fix the errors before running the Streamlit demo.")

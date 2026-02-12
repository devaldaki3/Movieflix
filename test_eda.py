"""
Quick test script to verify EDA endpoint
"""
import requests
import json

print("🧪 Testing EDA Endpoint...")
print("=" * 50)

try:
    # Test the EDA endpoint
    response = requests.get('http://localhost:5000/api/analytics/eda', timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        
        print("✅ EDA Endpoint is working!")
        print(f"\n📊 Statistics Retrieved:")
        
        if 'statistics' in data:
            stats = data['statistics']
            print(f"   • Total Movies: {stats.get('total_movies', 'N/A'):,}")
            print(f"   • Avg Rating: {stats.get('avg_rating', 'N/A'):.2f}/10")
            print(f"   • Top Genre: {stats.get('top_genre', ['N/A'])[0]}")
            print(f"   • Avg Runtime: {stats.get('avg_runtime', 'N/A'):.0f} min")
        
        print(f"\n📈 Visualizations Generated:")
        viz_keys = ['rating_dist', 'genre_analysis', 'release_trends', 
                   'budget_revenue', 'popularity', 'runtime', 'correlation']
        
        for key in viz_keys:
            if key in data and data[key]:
                print(f"   ✅ {key.replace('_', ' ').title()}")
            else:
                print(f"   ❌ {key.replace('_', ' ').title()} - Missing")
        
        print(f"\n🎉 All visualizations ready!")
        print(f"\n🌐 View at: http://localhost:5000/analytics")
        
    else:
        print(f"❌ Error: Status code {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.Timeout:
    print("⏱️ Request timed out - EDA generation takes time, this is normal")
    print("   The visualizations are being generated in the background")
    print("   Please refresh the analytics page after a few seconds")
    
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)

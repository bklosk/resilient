#!/usr/bin/env python3
import os, sys, json, requests
from geopy.geocoders import Photon
from geopy.distance import geodesic

def sqft_from_osm(lat, lon):
    q=f"[out:json];way['building'](around:50,{lat},{lon});out geom;"
    r=requests.get('https://overpass-api.de/api/interpreter',params={'data':q})
    if r.ok and r.json().get('elements'):
        g=r.json()['elements'][0]['geometry']
        l=[p['lat'] for p in g]; w=[p['lon'] for p in g]
        h=geodesic((min(l),min(w)),(max(l),min(w))).meters
        d=geodesic((min(l),min(w)),(min(l),max(w))).meters
        return h*d*10.7639

def main():
    addr=' '.join(sys.argv[1:]) or input('Address: ')
    loc=Photon(user_agent='rv').geocode(addr)
    sqft=sqft_from_osm(loc.latitude,loc.longitude) if loc else None
    sqft=sqft or 2300
    cost=float(os.getenv('COST_PER_SQFT',200))
    print(json.dumps({'address':addr,'sqft':sqft,'replacement_value':sqft*cost},indent=2))

if __name__=='__main__':
    main()

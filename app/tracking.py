"""A street map for a shipment that has not arrived.

Coordinates come from the order record. Tiles come from the OpenStreetMap tile server. The route and the
pulsing stop sit on that map.
"""

from __future__ import annotations

import html
import json

import streamlit as st


def render_tracking(tracking: dict | None) -> None:
    points = (tracking or {}).get("points") or []
    if not points:
        return
    st.iframe(_map_html(tracking), height=360)


def _map_html(tracking: dict) -> str:
    points = [
        {
            "lat": float(point["lat"]),
            "lng": float(point["lng"]),
            "place": str(point.get("place") or ""),
        }
        for point in tracking["points"]
    ]
    payload = json.dumps(points)
    place = html.escape(str(points[-1]["place"]))
    carrier = html.escape(str(tracking.get("carrier") or ""))
    caption = f"{place} · {carrier}" if carrier else place
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <link
          rel="stylesheet"
          href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
          html, body {{
            margin: 0;
            height: 100%;
            background: #fbf8f3;
            color: #1c1915;
            font-family: "Nunito Sans", "Segoe UI", sans-serif;
          }}
          .frame {{
            height: 100%;
            display: flex;
            flex-direction: column;
            border: 1px solid #e4ddd0;
            border-radius: 12px;
            overflow: hidden;
            background: #f4efe6;
          }}
          #map {{
            flex: 1;
            min-height: 280px;
            background: #e7e1d6;
          }}
          p {{
            margin: 0;
            padding: 0.45rem 0.75rem 0.55rem;
            font-size: 0.9rem;
            color: #5c564c;
          }}
          .pulse-wrap {{
            background: transparent;
            border: 0;
          }}
          .pulse {{
            width: 14px;
            height: 14px;
            margin: 2px;
            border-radius: 50%;
            background: #6e4c2f;
            box-shadow: 0 0 0 0 rgba(110, 76, 47, 0.55);
            animation: pulse 1.6s ease-out infinite;
          }}
          @keyframes pulse {{
            70% {{ box-shadow: 0 0 0 16px rgba(110, 76, 47, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(110, 76, 47, 0); }}
          }}
        </style>
      </head>
      <body>
        <div class="frame">
          <div id="map"></div>
          <p>{caption}</p>
        </div>
        <script>
          const points = {payload};
          const map = L.map("map", {{ zoomControl: true }});
          L.tileLayer("https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
            attribution: "&copy; OpenStreetMap",
            maxZoom: 19
          }}).addTo(map);
          const latlngs = points.map(function (point) {{ return [point.lat, point.lng]; }});
          L.polyline(latlngs, {{ color: "#1f3d34", weight: 3, opacity: 0.85 }}).addTo(map);
          points.slice(0, -1).forEach(function (point) {{
            L.circleMarker([point.lat, point.lng], {{
              radius: 5,
              color: "#1f3d34",
              fillColor: "#f4efe6",
              fillOpacity: 1,
              weight: 2
            }}).addTo(map);
          }});
          const last = points[points.length - 1];
          L.marker([last.lat, last.lng], {{
            icon: L.divIcon({{
              className: "pulse-wrap",
              html: '<div class="pulse"></div>',
              iconSize: [18, 18],
              iconAnchor: [9, 9]
            }})
          }}).addTo(map);
          if (latlngs.length === 1) {{
            map.setView(latlngs[0], 6);
          }} else {{
            map.fitBounds(latlngs, {{ padding: [36, 36] }});
          }}
          setTimeout(function () {{ map.invalidateSize(); }}, 150);
        </script>
      </body>
    </html>
    """

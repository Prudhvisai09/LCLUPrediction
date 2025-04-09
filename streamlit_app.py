import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# ------------------- Earth Engine Initialization ------------------- #
st.write("Initializing Earth Engine...")
ee.Authenticate()
ee.Initialize(project='ee-prudhviande17')

st.title("🌏 LULC Change Analysis & Prediction in Andhra Pradesh")
st.write("Select a district and click the Continue button to run the analysis.")

# ------------------- Sidebar District Selection ------------------- #
district_list = [
    "Anantapur", "Chittoor", "Cuddapah", "East Godavari", "Guntur", "Krishna",
    "Kurnool", "Nellore", "Prakasam", "Srikakulam", "Vishakhapatnam", "Vizianagaram",
    "West Godavari"
]
selected_district = st.selectbox("Select a District", district_list)

# ------------------- Continue Button ------------------- #
if st.button("Predict"):
    st.success(f"District selected: **{selected_district}**")

    # ------------------- Get ROI using GAUL dataset ------------------- #
    dataset = ee.FeatureCollection("FAO/GAUL/2015/level2")
    andhraPradesh = dataset.filter(ee.Filter.eq('ADM1_NAME', 'Andhra Pradesh'))
    roi = andhraPradesh.filter(ee.Filter.eq('ADM2_NAME', selected_district)).geometry()

    # ------------------- Define Dynamic World Variables ------------------- #
    dw_palette = ['419BDF', '397D49', '88B053', '7A87C6', 'E49635',
                  'DFC35A', 'C4281B', 'A59B8F', 'B39FE1']
    dw_names = ['Water', 'Trees', 'Grass', 'Flooded Vegetation',
                'Crops', 'Shrub & Scrub', 'Built Area', 'Bare Ground', 'Snow & Ice']

    # Function to get the modal (mode) land cover image for a given year.
    def get_dw_mode(year):
        collection = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
            .filterBounds(roi) \
            .filterDate(f'{year}-01-01', f'{year}-12-31') \
            .select('label')
        return collection.mode().clip(roi).toInt()

    # ------------------- Load LULC Images ------------------- #
    lulc_2020 = get_dw_mode(2020).rename('start')
    lulc_2024 = get_dw_mode(2024).rename('end')

    # Display Map1 with LULC 2020 and 2024, centering on the ROI
    Map = geemap.Map()
    Map.centerObject(roi, zoom=8)
    Map.addLayer(lulc_2020, {'min': 0, 'max': 8, 'palette': dw_palette}, 'LULC 2020')
    Map.addLayer(lulc_2024, {'min': 0, 'max': 8, 'palette': dw_palette}, 'LULC 2024')
    Map.add_legend(title="Land Cover", labels=dw_names, colors=dw_palette)
    st.subheader("Interactive Map of LULC 2020 & 2024")
    Map.to_streamlit(height=500)

    # ------------------- Create Transition Map ------------------- #
    transition_map = lulc_2020.multiply(10).add(lulc_2024).rename('transition')
    change_only = transition_map.updateMask(lulc_2020.neq(lulc_2024))
    Map.addLayer(change_only, {'min': 0, 'max': 88, 'palette': dw_palette * 2}, 'LULC Change 2020-2024')

    # ------------------- Load Elevation and Create Year Band ------------------- #
    elevation = ee.Image("USGS/SRTMGL1_003").clip(roi).rename('elevation')
    year_band = ee.Image.constant(2024).rename('year')

    # ------------------- Stack Training Variables & Create Training Image ------------------- #
    training_vars = lulc_2020 \
        .addBands(transition_map.rename('transition')) \
        .addBands(elevation) \
        .addBands(year_band)
    label = lulc_2024.rename('end')
    training_image = training_vars.addBands(label)

    # ------------------- Create Training and Testing Samples ------------------- #
    sample = training_image.stratifiedSample(
        numPoints=3000,
        classBand='transition',
        scale=10,
        region=roi,
        seed=42,
        geometries=False
    ).randomColumn()
    train = sample.filter(ee.Filter.lte('random', 0.8))
    test = sample.filter(ee.Filter.gt('random', 0.8))
    st.write("✅ Training and testing samples created.")

    # ------------------- Train Random Forest Classifier ------------------- #
    feature_names = ['start', 'transition', 'elevation', 'year']
    label_name = 'end'
    classifier = ee.Classifier.smileRandomForest(numberOfTrees=100) \
        .train(features=train, classProperty=label_name, inputProperties=feature_names)



    # ------------------- Predict for 2028 (Modified Water-to-Built) ------------------- #
    year_2028 = ee.Image.constant(2028).rename('year')
    variables_2028 = lulc_2024.rename('start') \
        .addBands(transition_map.rename('transition')) \
        .addBands(elevation) \
        .addBands(year_2028)
    lulc_2028 = variables_2028.classify(classifier).rename('LULC_2028')
    w_start = lulc_2024.eq(0)
    w_pred = lulc_2028.eq(0)   
    s_start = lulc_2024.eq(8)  
    s_pred = lulc_2028.eq(8)   
    # Apply subtle transition rules
    lulc_2028_modified = lulc_2028 \
        .where(w_start, 0) \
        .where(w_pred.And(w_start.Not()), 6) \
        .where(s_start, 8) \
        .where(s_pred.And(s_start.Not()), 5)

    Map2 = geemap.Map()
    Map2.centerObject(roi, zoom=8)
    Map2.addLayer(lulc_2020, {'min': 0, 'max': 8, 'palette': dw_palette}, 'LULC 2020')
    Map2.addLayer(lulc_2024, {'min': 0, 'max': 8, 'palette': dw_palette}, 'LULC 2024')
    Map2.addLayer(lulc_2028_modified, {'min': 0, 'max': 8, 'palette': dw_palette}, 'Predicted LULC 2028')
    Map2.add_legend(title="Land Cover", labels=dw_names, colors=dw_palette)
    st.subheader("Map 2: 2020, 2024 & Predicted LULC for 2028")
    Map2.to_streamlit(height=500)

    # ------------------- Compute Area by Class ------------------- #
    def compute_area(image, year):
        area_img = ee.Image.pixelArea().divide(10000).addBands(image.rename('class'))
        stats = area_img.reduceRegion(
            reducer=ee.Reducer.sum().group(1, 'class'),
            geometry=roi,
            scale=10,
            bestEffort=True
        )
        areas = ee.List(stats.get('groups'))
        fc = ee.FeatureCollection(areas.map(lambda item: ee.Feature(None, {
            'year': year,
            'class': ee.Dictionary(item).get('class'),
            'area_ha': ee.Dictionary(item).get('sum')
        })))
        return fc

    areas_2020 = compute_area(lulc_2020, 2020)
    areas_2024 = compute_area(lulc_2024, 2024)
    areas_2028 = compute_area(lulc_2028_modified, 2028)
    lulc_areas = areas_2020.merge(areas_2024).merge(areas_2028)

    lulc_area_info = lulc_areas.getInfo()
    lulc_df = pd.DataFrame([f['properties'] for f in lulc_area_info['features']])
    class_labels_map = {
        0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Veg',
        4: 'Crops', 5: 'Shrub & Scrub', 6: 'Built', 7: 'Bare', 8: 'Snow/Ice'
    }
    lulc_df['class'] = lulc_df['class'].map(class_labels_map)

    # ------------------- Create Pivot Table ------------------- #
    pivot_table = lulc_df.pivot_table(
        index='class',
        columns='year',
        values='area_ha',
        aggfunc='sum'
    ).fillna(0).round(2)

    st.write("### Pivot Table of Area (ha)")
    st.dataframe(pivot_table)

    # ------------------- PLOT 1: Interactive Altair Line Chart ------------------- #
    st.write("### Interactive Altair Line Chart")
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['year'], empty='none')
    base_line = alt.Chart(lulc_df).mark_line(interpolate='basis').encode(
        x='year:Q',
        y='area_ha:Q',
        color='class:N'
    )
    selectors = alt.Chart(lulc_df).mark_point().encode(
        x='year:Q',
        opacity=alt.value(0)
    ).add_selection(nearest)
    points = base_line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    text = base_line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'area_ha:Q', alt.value(' '))
    )
    rules = alt.Chart(lulc_df).mark_rule(color='gray').encode(
        x='year:Q'
    ).transform_filter(nearest)
    alt_chart = alt.layer(base_line, selectors, points, rules, text).properties(width=600, height=300)
    st.altair_chart(alt_chart, use_container_width=True)

    # ------------------- PLOT 2: Stacked Area Plot ------------------- #
    st.write("### Stacked Area Plot")
    fig_area, ax_area = plt.subplots(figsize=(12, 6))
    pivot_t = pivot_table.T
    pivot_t.plot.area(ax=ax_area)
    ax_area.set_title('LULC Area Change Over Time (Stacked Area)')
    ax_area.set_xlabel('Year')
    ax_area.set_ylabel('Area (ha)')
    ax_area.legend(title='LULC Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig_area)

    # ------------------- PLOT 3: Line Chart with Markers (Matplotlib) ------------------- #
    st.write("### Line Chart by LULC Class (Matplotlib)")
    fig_line, ax_line = plt.subplots(figsize=(12, 6))
    for c in pivot_table.index:
        data = pivot_table.loc[c]
        ax_line.plot(data.index, data.values, marker='o', label=c)
    ax_line.set_title('LULC Trend Over Time by Class')
    ax_line.set_xlabel('Year')
    ax_line.set_ylabel('Area (ha)')
    ax_line.grid(True)
    ax_line.legend(title='LULC Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig_line)

    # ------------------- PLOT 4: Grouped Bar Plot ------------------- #
    st.write("### Grouped Bar Plot")
    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
    pivot_table[[2020, 2024, 2028]].plot(kind='bar', ax=ax_bar)
    ax_bar.set_title('LULC Area Comparison in Selected Years')
    ax_bar.set_xlabel('LULC Class')
    ax_bar.set_ylabel('Area (ha)')
    ax_bar.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig_bar)

    # ------------------- PLOT 5: Heatmap ------------------- #
    st.write("### Heatmap of LULC Areas")
    fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap='YlGnBu', ax=ax_heat)
    ax_heat.set_title("Heatmap of LULC Area (ha)")
    st.pyplot(fig_heat)

    # ------------------- PLOT 7: Pie Charts for Each Year ------------------- #
    st.write("### Pie Charts of LULC Share by Year")
    for yr in [2020, 2024, 2028]:
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        series = pivot_table[yr]
        ax_pie.pie(series, labels=series.index, autopct='%1.1f%%', startangle=90)
        ax_pie.set_title(f'LULC Share in {yr}')
        st.pyplot(fig_pie)

    st.success("All plots are generated on Streamlit!")
    # ------------------- Classify Test Set for Accuracy with Progress Indicator ------------------- #

    test_classified = test.classify(classifier, 'prediction')
    conf_matrix = test_classified.errorMatrix(label_name, 'prediction')
    st.write("✅ Confusion Matrix:", conf_matrix.getInfo())
    st.write("🎯 Accuracy:", conf_matrix.accuracy().getInfo())
    st.write("📈 Kappa:", conf_matrix.kappa().getInfo())

# Первичные источники

Проверены при подготовке проекта 17 сентября 2026 года. Ссылки описывают продукты,
а фактическую доступность по району нужно проверять запросами к каталогу.

| Источник | Что проверять |
|---|---|
| [ESA Sentinel-1 products](https://sentiwiki.copernicus.eu/web/s1-products) | L1 GRD/SLC; L2 OCN — параметры океана, режимы и разрешение |
| [ESA Sentinel-2 products](https://sentiwiki.copernicus.eu/web/s2-products) | L2A surface reflectance, каналы, SCL и ограничения |
| [ESA Sentinel-2 mission](https://sentiwiki.copernicus.eu/web/s2-mission) | Планирование съёмки, наземные/прибрежные районы |
| [Element84 Earth Search](https://github.com/Element84/earth-search) | STAC, COG, per-asset scale/offset и известные пробелы каталога |
| [CDSE OData](https://documentation.dataspace.copernicus.eu/APIs/OData.html) | Поиск и загрузка Sentinel |
| [CDSE token](https://documentation.dataspace.copernicus.eu/APIs/Token.html) | Бесплатная регистрация и токен загрузки |
| [USGS Landsat C2 L2](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) | Surface reflectance и QA |
| [USGS ограничения SR](https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance) | Высокие широты и низкое солнце |
| [NSIDC VNP29 v2](https://nsidc.org/data/vnp29/versions/2) | L2 sea ice swath, 375 м, Earthdata Login |
| [NSIDC MOD29 v61](https://nsidc.org/data/mod29/versions/61) | L2 swath, 1 км, Earthdata Login |
| [Исследование grounded ridges, 2025](https://tc.copernicus.org/articles/19/2045/2025/) | Заземление, глубина и независимые высотные наблюдения |
| [Nansen sea_ice_drift](https://github.com/nansencenter/sea_ice_drift) | Исследовательский SAR feature/pattern tracking; GPL-3.0 |
| [SNAP Ellipsoid Correction](https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-toolboxes/org.esa.s1tbx.s1tbx.op.sar.processing.ui/operators/EllipsoidCorrectionRDOp.html) | Геокодирование с усреднённой высотой сцены |

Репозитории найдены/проверены через подключённый GitHub. Код `sea_ice_drift`
в этот проект не скопирован; его лицензия требует отдельного решения при интеграции.
Открытые данные не гарантируют доступ без аккаунта, без квот или без задержки.
Стоимость вычислений/хранения не равна стоимости самих спутниковых продуктов.

Сохранять требуемые атрибуции поставщика при распространении результатов, включая
Copernicus Sentinel data с годом съёмки и DOI NASA-продуктов при их подключении.

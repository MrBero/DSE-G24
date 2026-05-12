from ansys.fluent.core import launch_fluent
import re
s = launch_fluent(product_version="26.1.0",
                  mode="solver",
                  precision="double",
                  dimension=2)


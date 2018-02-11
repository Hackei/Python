"""
#ROW XML DATA
<?xml version="1.0"?>
<data>
    <country name="Liechtenstein">
        <rank>1</rank>
        <year>2008</year>
        <gdppc>141100</gdppc>
        <neighbor name="Austria" direction="E"/>
        <neighbor name="Switzerland" direction="W"/>
    </country>
    <country name="Singapore">
        <rank>4</rank>
        <year>2011</year>
        <gdppc>59900</gdppc>
        <neighbor name="Malaysia" direction="N"/>
    </country>
    <country name="Panama">
        <rank>68</rank>
        <year>2011</year>
        <gdppc>13600</gdppc>
        <neighbor name="Costa Rica" direction="W"/>
        <neighbor name="Colombia" direction="E"/>
	<newTag>
	  <newTag2>"aaaa"</newTag2>
	  <newTag3>"bbbb"</newTag3>
	</newTag>
    </country>
</data>
"""

from xml.etree import ElementTree as ET

tree = ET.parse("test.xml")#file_name

for c in tree.iter():
    idx = 0
    if c.tag == "country":
        newParent = ET.Element("refInfo")
        c.append(newParent)
        #newParent = c.append(newParent)
        #c.append(newParent)#Create 'refInfo' element(parent)
        newChildSys = ET.SubElement(newParent,"SysName")
        newChildTime = ET.SubElement(newParent,"Time")
        newChildSys.text = "SysName"
        newChildTime.text = "UpdatedTime"
                         
tree.write("output.xml")

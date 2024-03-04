import { Image, FlatList,Pressable, StyleSheet, Text, View } from "react-native";
function FilesList(props){
    return(
        <View style={styles.subjects}>
    <Pressable android_ripple={{color:'#dddddd'}}>
        <Text style={styles.subjects}>{props.text}</Text>
    </Pressable>
      </View>
    )
}

export default FilesList;

const styles = StyleSheet.create({
    
    subjects: {
      backgroundColor: "#344B79",
      margin: 10,
      padding: 20,
      borderRadius: 9,
      width: 350,
      height: 100,
      alignItems: "center",
      color: "white",
      justifyContent: "center",
    },
  });
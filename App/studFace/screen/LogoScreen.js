import { useEffect } from "react";
import { View, Image, StyleSheet } from "react-native";

function LogoScreen({ navigation }) {
  useEffect(()=>{
      const timer=setTimeout(()=>{
          navigation.navigate('Login');
      },2000)
      return ()=>clearTimeout(timer);
  },[]);

  return (
    <View style={styles.imageContainer}>
      <Image style={styles.image} source={require("../assets/images/logo.png")} />
    </View>
  );
}

export default LogoScreen;

LogoScreen.navigationOptions = {
  headerShown: false,
  headerTransparent: true,
};

const styles = StyleSheet.create({
  imageContainer: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    width: 400,
  },
  image:{
    height:'100%'
  }
});

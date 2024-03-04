import React from "react";
import { TouchableOpacity, Text, StyleSheet } from "react-native";

function SubjectsList({ text, onPress }) {
  return (
    <TouchableOpacity style={styles.subjectContainer} onPress={onPress}>
      <Text style={styles.subjectText}>{text}</Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  subjectContainer: {
    backgroundColor: "#344B79",
    margin: 10,
    padding: 20,
    borderRadius: 9,
    width: 350,
    height: 100,
    alignItems: "center",
    justifyContent: "center",
  },
  subjectText: {
    color: "white",
    fontSize: 16,
  },
});

export default SubjectsList;

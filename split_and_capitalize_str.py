def splitncap(str, splitter = "_"):
  x = str.split(splitter)
  return(" ".join(x).capitalize())

if __name__ == "__main__":
  print(splitncap("name_surname", "_"))
  print(splitncap("name surname", " "))

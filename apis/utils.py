
def Custom_cap_function_fast(custon_json):
    # === Unpack input ===
    # age_val = int(custon_json['age'])
    genders = custon_json['gender']
    spoken_style = custon_json['spoken_style']
    emotion_style = custon_json['emotion_style']

    emotion = custon_json['emotion']
    intensity = custon_json['intensity']
    intensity_array=['','"약하게"','"중간정도의"','"강하게"']
    #speed = custon_json['pace']

    # if age
    if genders == "MALE":
        gender = '"남성"의 목소리로'
    elif genders == "FEMALE":
        gender = '"여성"의 목소리로'
    else: ### will be fix (단로로운 -> 단조로운)
        gender = ''
    if emotion_style:
        if spoken_style:
            if emotion=='무감정':
                emotion='"중립적인" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='기쁨':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+ ' "기쁜" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='슬픔':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "슬픈" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='분노':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "분노하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='당황':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "당황하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='불안':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "불안한" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='상처':
                emotion='"' + emotion_style + '"'+' 느낌의 '+intensity_array[intensity]+' "상처받은" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
        else:
            if emotion=='무감정':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"중립적인" 감정'
            elif emotion=='기쁨':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"기쁜" 감정'
            elif emotion=='슬픔':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"슬픈" 감정'
            elif emotion=='분노':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"분노하는" 감정'
            elif emotion=='당황':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"당황하는" 감정'
            elif emotion=='불안':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"불안한" 감정'
            elif emotion=='상처':
                emotion='"' + emotion_style + '"'+' 느낌의 '+'"상처받은" 감정'
    else:
        if intensity_array:
            if emotion=='무감정':
                emotion='"중립적인" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='기쁨':
                emotion=intensity_array[intensity]+ ' "기쁜" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='슬픔':
                emotion=intensity_array[intensity]+' "슬픈" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='분노':
                emotion=intensity_array[intensity]+' "분노하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='당황':
                emotion=intensity_array[intensity]+' "당황하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='불안':
                emotion=intensity_array[intensity]+' "불안한" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='상처':
                emotion=intensity_array[intensity]+' "상처받은" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
        else:
            if emotion=='무감정':
                emotion='"중립적인" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='기쁨':
                emotion='"기쁜" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='슬픔':
                emotion='"슬픈" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='분노':
                emotion='"분노하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='당황':
                emotion='"당황하는" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='불안':
                emotion='"불안한" 감정이고 '+'"'+spoken_style+'"'+' 스타일'
            elif emotion=='상처':
                emotion='"상처받은" 감정이고 '+'"'+spoken_style+'"'+' 스타일'

    parts = []

    if emotion: parts.append(emotion)

    caption=' '.join(parts)
    # === Final composition ===

    caption=emotion+'<|endofprompt|>'
    # === Final composition ===
    return caption


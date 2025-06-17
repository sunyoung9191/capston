
package com.example.beautyinside;

import java.io.Serializable;

public class ReviewData implements Serializable {
    private String beforeImage;
    private String afterImage;
    private String clinicName;
    private String doctorName;
    private String procedureType;
    private String hospitalCode;

    private String userName;
    private String type;
    private String text;

    // 기존 8개 인자 생성자
    public ReviewData(String userName, String type, String text, String beforeImage, String afterImage, String doctorName, String clinicName, String procedureType) {
        this.userName = userName;
        this.type = type;
        this.text = text;
        this.beforeImage = beforeImage;
        this.afterImage = afterImage;
        this.doctorName = doctorName;
        this.clinicName = clinicName;
        this.procedureType = procedureType;
    }

    // ✅ 새로운 6개 인자용 생성자 (hospitalCode, text 없는 경우용)
    public ReviewData(String userName, String type, String text, String beforeImage, String afterImage, String doctorName) {
        this.userName = userName;
        this.type = type;
        this.text = text;
        this.beforeImage = beforeImage;
        this.afterImage = afterImage;
        this.doctorName = doctorName;
        this.clinicName = "";
        this.procedureType = "";
    }

    public String getBeforeImage() {
        return beforeImage;
    }

    public String getAfterImage() {
        return afterImage;
    }

    public String getClinicName() {
        return clinicName;
    }

    public String getDoctorName() {
        return doctorName;
    }

    public String getProcedureType() {
        return procedureType;
    }

    public String getHospitalCode() {
        return hospitalCode;
    }

    public void setBeforeImage(String beforeImage) {
        this.beforeImage = beforeImage;
    }

    public void setAfterImage(String afterImage) {
        this.afterImage = afterImage;
    }

    public void setClinicName(String clinicName) {
        this.clinicName = clinicName;
    }

    public void setDoctorName(String doctorName) {
        this.doctorName = doctorName;
    }

    public void setProcedureType(String procedureType) {
        this.procedureType = procedureType;
    }

    public void setHospitalCode(String hospitalCode) {
        this.hospitalCode = hospitalCode;
    }

    public String getUserName() {
        return userName;
    }

    public String getType() {
        return type;
    }

    public String getText() {
        return text;
    }
}

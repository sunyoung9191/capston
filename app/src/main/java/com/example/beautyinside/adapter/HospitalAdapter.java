package com.example.beautyinside.adapter;

import android.content.Context;
import android.content.Intent;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import com.example.beautyinside.R;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import java.io.Serializable;

import java.util.List;
import com.example.beautyinside.HospitalData;
import com.example.beautyinside.hospital.AmondHospitalDetailActivity;
import com.example.beautyinside.hospital.BabaHospitalDetailActivity;
import com.example.beautyinside.hospital.ClasyHospitalDetailActivity;
import com.example.beautyinside.hospital.DmHospitalDetailActivity;
import com.example.beautyinside.hospital.DressHospitalDetailActivity;
import com.example.beautyinside.hospital.EsHospitalDetailActivity;
import com.example.beautyinside.hospital.HitHospitalDetailActivity;
import com.example.beautyinside.hospital.MadeHospitalDetailActivity;
import com.example.beautyinside.hospital.MarbleHospitalDetailActivity;
import com.example.beautyinside.hospital.MindHospitalDetailActivity;
import com.example.beautyinside.hospital.MlHospitalDetailActivity;
import com.example.beautyinside.hospital.NanaHospitalDetailActivity;
import com.example.beautyinside.hospital.OnairHospitalDetailActivity;
import com.example.beautyinside.hospital.PopHospitalDetailActivity;
import com.example.beautyinside.hospital.RugoHospitalDetailActivity;
import com.example.beautyinside.hospital.ThankyouHospitalDetailActivity;
import com.example.beautyinside.hospital.TsHospitalDetailActivity;
import com.example.beautyinside.hospital.WonderHospitalDetailActivity;
import com.example.beautyinside.hospital.YellowHospitalDetailActivity;
import com.example.beautyinside.hospital.YounoHospitalDetailActivity;
import com.example.beautyinside.FavoriteManager;
import com.example.beautyinside.MainActivity;
import android.widget.Toast;
import android.util.Log;
import java.util.ArrayList;
public class HospitalAdapter extends RecyclerView.Adapter<HospitalAdapter.ViewHolder> {

    private Context context;
    private List<HospitalData> hospitalList;

    public HospitalAdapter(Context context, List<HospitalData> hospitalList) {
        this.context = context;
        this.hospitalList = hospitalList != null ? hospitalList : new ArrayList<>(); // null 방지
    }

    // 데이터 업데이트 메서드 추가
    public void updateData(List<HospitalData> newList) {
        this.hospitalList = newList != null ? newList : new ArrayList<>();
        notifyDataSetChanged();
    }

    @NonNull
    @Override
    public HospitalAdapter.ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(context).inflate(R.layout.item_hospital_card, parent, false);
        return new ViewHolder(view);
    }
    public void setFilteredList(List<HospitalData> newList) {
        this.hospitalList = newList;
        notifyDataSetChanged();
    }

    @Override
    public void onBindViewHolder(@NonNull HospitalAdapter.ViewHolder holder, int position) {
        // 인덱스 범위 체크 추가
        if (position < 0 || position >= hospitalList.size()) {
            return;
        }

        HospitalData hospital = hospitalList.get(position);

        if (holder.hospitalName != null) {
            holder.hospitalName.setText(hospital.getName());
        }
        if (hospital == null) {
            return; // null 체크 추가
        }

        holder.hospitalName.setText(hospital.getName());
        holder.bannerImage.setImageResource(hospital.getBannerImageResId());

        // 병원 상세페이지로 이동 (수정)
        holder.itemView.setOnClickListener(v -> {
            try {
                String hospitalName = hospital.getName();
                Class<?> detailClass = getDetailActivityClass(hospitalName);

                Intent intent = new Intent(context, detailClass);
                intent.putExtra("hospitalName", hospitalName);

                // HospitalData 객체를 넘기려면 HospitalData 클래스에 implements Serializable 필요!
                if (hospital instanceof Serializable) {
                    intent.putExtra("hospital_data", hospital);
                } else {
                    Log.w("HospitalAdapter", "HospitalData가 Serializable이 아님");
                }

                context.startActivity(intent);
            } catch (Exception e) {
                Log.e("HospitalAdapter", "상세페이지 이동 중 오류: " + e.getMessage(), e);
                Toast.makeText(context, "페이지를 열 수 없어요 ㅠㅠ", Toast.LENGTH_SHORT).show();
            }
        });

        // 찜 버튼 (try-catch 추가)
        try {
            boolean isFavorite = FavoriteManager.getInstance().isFavorite(hospital);
            updateFavoriteIcon(holder.favoriteButton, isFavorite);

            holder.favoriteButton.setOnClickListener(v -> {
                boolean nowFavorite = !FavoriteManager.getInstance().isFavorite(hospital);
                if (nowFavorite) {
                    FavoriteManager.getInstance().addFavorite(hospital);
                    Toast.makeText(context, "찜 목록에 추가됐어요!", Toast.LENGTH_SHORT).show();
                } else {
                    FavoriteManager.getInstance().removeFavorite(hospital);
                    Toast.makeText(context, "찜 목록에서 삭제됐어요!", Toast.LENGTH_SHORT).show();
                }
                updateFavoriteIcon(holder.favoriteButton, nowFavorite);

                // 찜 목록 화면 갱신을 위한 이벤트 발생 (EventBus 사용 시)
                // EventBus.getDefault().post(new FavoriteUpdateEvent());
            });
        } catch (Exception e) {
            Log.e("HospitalAdapter", "찜 버튼 처리 중 오류: " + e.getMessage());
        }
    }

    private void updateFavoriteIcon(ImageButton button, boolean isFavorite) {
        if (button == null) return; // null 체크 추가

        if (isFavorite) {
            button.setImageResource(R.drawable.ic_heart_filled);
        } else {
            button.setImageResource(R.drawable.ic_heart_border);
        }
    }

    private Class<?> getDetailActivityClass(String hospitalName) {
        if (hospitalName == null) {
            return MainActivity.class; // null 체크 추가
        }

        // 병원이름에 따라 상세 페이지 Activity 클래스 연결
        switch (hospitalName) {
            case "바바 성형외과":
                return BabaHospitalDetailActivity.class;
            case "아몬드 성형외과":
                return AmondHospitalDetailActivity.class;
            case "나나 성형외과":
                return NanaHospitalDetailActivity.class;
            case "팝 성형외과":
                return PopHospitalDetailActivity.class;
            case "티에스 성형외과":
                return TsHospitalDetailActivity.class;
            case "루호 성형외과":
                return RugoHospitalDetailActivity.class;
            case "마인드 성형외과":
                return MindHospitalDetailActivity.class;
            case "드레스 성형외과":
                return DressHospitalDetailActivity.class;
            case "디엠 성형외과":
                return DmHospitalDetailActivity.class;
            case "옐로우 성형외과":
                return YellowHospitalDetailActivity.class;
            case "히트 성형외과":
                return HitHospitalDetailActivity.class;
            case "마블 성형외과":
                return MarbleHospitalDetailActivity.class;
            case "온에어 성형외과":
                return OnairHospitalDetailActivity.class;
            case "유노 성형외과":
                return YounoHospitalDetailActivity.class;
            case "메이드영 성형외과":
                return MadeHospitalDetailActivity.class;
            case "땡큐 성형외과":
                return ThankyouHospitalDetailActivity.class;
            case "원더풀 성형외과":
                return WonderHospitalDetailActivity.class;
            case "이에스 성형외과":
                return EsHospitalDetailActivity.class;
            case "일미리 성형외과":
                return MlHospitalDetailActivity.class;
            case "클래시 성형외과":
                return ClasyHospitalDetailActivity.class;
            default:
                Log.w("HospitalAdapter", "알 수 없는 병원 이름: " + hospitalName);
                return MainActivity.class;
        }
    }

    @Override
    public int getItemCount() {
        return hospitalList != null ? hospitalList.size() : 0; // null 체크 추가
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        TextView hospitalName;
        ImageView bannerImage;
        ImageButton favoriteButton;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);
            hospitalName = itemView.findViewById(R.id.hospital_name);
            bannerImage = itemView.findViewById(R.id.bannerImageView);
            favoriteButton = itemView.findViewById(R.id.buttonFavorite);
        }
    }
}

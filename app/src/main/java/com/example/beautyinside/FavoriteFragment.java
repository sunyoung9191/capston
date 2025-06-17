package com.example.beautyinside;

import android.os.Bundle;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import java.util.List;
import com.example.beautyinside.adapter.HospitalAdapter;
public class FavoriteFragment extends Fragment {

    private RecyclerView recyclerView;
    private HospitalAdapter adapter;

    public FavoriteFragment() {
        // Required empty public constructor
    }

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        View view = inflater.inflate(R.layout.fragment_favorite, container, false);

        recyclerView = view.findViewById(R.id.recyclerFavorites);
        recyclerView.setLayoutManager(new LinearLayoutManager(getContext()));

        List<HospitalData> favoriteList = FavoriteManager.getInstance().getFavoriteList();

        adapter = new HospitalAdapter(getContext(), favoriteList);
        recyclerView.setAdapter(adapter);

        return view;
    }
}

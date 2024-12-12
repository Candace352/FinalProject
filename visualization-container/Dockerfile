
FROM grafana/grafana:latest
WORKDIR /etc/grafana
COPY grafana.ini /etc/grafana/grafana.ini
RUN grafana-cli plugins install grafana-piechart-panel
EXPOSE 3000
CMD ["/run.sh"]


#!/bin/bash
#SBATCH --job-name=mars_request
#SBATCH --qos=nf
#SBATCH --time=45:00:00
#SBATCH --mem=128G
#SBATCH --output=/home/ecm5702/dev/post_prepml/tc/events/out/mars_retrieve.%j.out
#SBATCH --error=/home/ecm5702/dev/post_prepml/tc/events/out/mars_error.%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=%USER%@ecmwf.int

EXPID="j24v"


# IDALIA


AREA="40/-100/10/-70"
GRID="O320"
STREAM="ENFO"
TYPE="PF"
CLASS="RD"
STEP="24/to/120/by/24"
PARAMS="151.128/165.128/166.128/167.128"
NUMBER="1/to/10/by/1"
TIMES="0000"
YEAR="2023"
MONTH="08"
#for ML experiments used FDB retrievalsb and add DATABASE variables in RETRIEVE below
DATABASE="FDB"
LEVEL="SFC"
DAYS="26 27 28 29 30"


for y in ${YEAR}; do
 
  for m in ${MONTH}; do
    #get the number of days for this particular month/year
    days_per_month=$(cal ${m} ${y} | awk 'NF {DAYS = $NF}; END {print DAYS}')
     
    #date loop
    for my_date in ${DAYS}; do
      my_date=${YEAR}${m}${my_date}
      echo $my_date
       
      #time lop
      for my_time in ${TIMES}; do
        cat << EOF > my_request_${my_date}_${my_time}.mars

RETRIEVE,
    CLASS      = ${CLASS},
    TYPE       = ${TYPE},
    STREAM     = ${STREAM},
    EXPVER     = ${EXPID},
    LEVTYPE    = ${LEVEL},
    AREA       = ${AREA},
    PARAM      = ${PARAMS},
    DATE       = ${my_date},
    TIME       = ${my_time},
    NUMBER     = ${NUMBER},
    GRID       = ${GRID},
    DATABASE   = ${DATABASE},
    STEP       = ${STEP},
    TARGET     = "/home/ecm5702/perm/reference/o96_o320/tc/idalia/surface_pf_${STREAM}_${GRID}_${EXPID}_${my_date}.grib"
EOF
      mars my_request_${my_date}_${my_time}.mars
      status=$?

      sync
      sleep 5

      if [ $status -eq 0 ]; then
        rm -f my_request_${my_date}_${my_time}.mars
      fi
      done
    done
  done
done


## HILARY

AREA="35/-125/0/-95"
GRID="O320"

STREAM="ENFO"
TYPE="PF"
CLASS="RD"
STEP="24/to/120/by/24"
PARAMS="151.128/165.128/166.128/167.128"
NUMBER="1/to/10/by/1"
TIMES="0000"
YEAR="2023"
MONTH="08"
DATABASE="FDB"
LEVEL="SFC"
DAYS="16 17 18 19 20"


for y in ${YEAR}; do
 
  for m in ${MONTH}; do
    #get the number of days for this particular month/year
    days_per_month=$(cal ${m} ${y} | awk 'NF {DAYS = $NF}; END {print DAYS}')
     
    #date loop
    for my_date in ${DAYS}; do
      my_date=${YEAR}${m}${my_date}
      echo $my_date
       
      #time lop
      for my_time in ${TIMES}; do
        cat << EOF > my_request_${my_date}_${my_time}.mars

RETRIEVE,
    CLASS      = ${CLASS},
    TYPE       = ${TYPE},
    STREAM     = ${STREAM},
    EXPVER     = ${EXPID},
    LEVTYPE    = ${LEVEL},
    AREA       = ${AREA},
    PARAM      = ${PARAMS},
    DATE       = ${my_date},
    TIME       = ${my_time},
    NUMBER     = ${NUMBER},
    GRID       = ${GRID},
    DATABASE   = ${DATABASE},
    STEP       = ${STEP},
    TARGET     = "/home/ecm5702/perm/reference/o96_o320/tc/hilary/surface_pf_${STREAM}_${GRID}_${EXPID}_${my_date}.grib"
EOF
      mars my_request_${my_date}_${my_time}.mars
      status=$?

      sync
      sleep 5

      if [ $status -eq 0 ]; then
        rm -f my_request_${my_date}_${my_time}.mars
      fi
      done
    done
  done
done

sync
sleep 5

## FRANKLIN

AREA="40/-80/10/-50"
GRID="O320"
STREAM="ENFO"
TYPE="PF"
CLASS="RD"
STEP="24/to/120/by/24"
PARAMS="151.128/165.128/166.128/167.128"
NUMBER="1/to/10/by/1"
TIMES="0000"
YEAR="2023"
MONTH="08"
DATABASE="FDB"
LEVEL="SFC"
DAYS="20 21 22 23 24 25 26 27 28 29 30"


for y in ${YEAR}; do
 
  for m in ${MONTH}; do
    #get the number of days for this particular month/year
    days_per_month=$(cal ${m} ${y} | awk 'NF {DAYS = $NF}; END {print DAYS}')
     
    #date loop
    for my_date in ${DAYS}; do
      my_date=${YEAR}${m}${my_date}
      echo $my_date
       
      #time lop
      for my_time in ${TIMES}; do
        cat << EOF > my_request_${my_date}_${my_time}.mars

RETRIEVE,
    CLASS      = ${CLASS},
    TYPE       = ${TYPE},
    STREAM     = ${STREAM},
    EXPVER     = ${EXPID},
    LEVTYPE    = ${LEVEL},
    AREA       = ${AREA},
    PARAM      = ${PARAMS},
    DATE       = ${my_date},
    TIME       = ${my_time},
    NUMBER     = ${NUMBER},
    GRID       = ${GRID},
    DATABASE   = ${DATABASE},
    STEP       = ${STEP},
    TARGET     = "/home/ecm5702/perm/reference/o96_o320/tc/franklin/surface_pf_${STREAM}_${GRID}_${EXPID}_${my_date}.grib"
EOF
      mars my_request_${my_date}_${my_time}.mars
      status=$?

      sync
      sleep 5

      if [ $status -eq 0 ]; then
        rm -f my_request_${my_date}_${my_time}.mars
      fi
      done
    done
  done
done


## FERNANDA

AREA="30/-135/0/-105"
GRID="O320"
STREAM="ENFO"
TYPE="PF"
CLASS="RD"
STEP="24/to/120/by/24"
PARAMS="151.128/165.128/166.128/167.128"
NUMBER="1/to/10/by/1"
TIMES="0000"
YEAR="2023"
MONTH="08"
DATABASE="FDB"
LEVEL="SFC"
DAYS="12 13 14 15 16 17"


for y in ${YEAR}; do
 
  for m in ${MONTH}; do
    #get the number of days for this particular month/year
    days_per_month=$(cal ${m} ${y} | awk 'NF {DAYS = $NF}; END {print DAYS}')
     
    #date loop
    for my_date in ${DAYS}; do
      my_date=${YEAR}${m}${my_date}
      echo $my_date
       
      #time lop
      for my_time in ${TIMES}; do
        cat << EOF > my_request_${my_date}_${my_time}.mars

RETRIEVE,
    CLASS      = ${CLASS},
    TYPE       = ${TYPE},
    STREAM     = ${STREAM},
    EXPVER     = ${EXPID},
    LEVTYPE    = ${LEVEL},
    AREA       = ${AREA},
    PARAM      = ${PARAMS},
    DATE       = ${my_date},
    TIME       = ${my_time},
    NUMBER     = ${NUMBER},
    GRID       = ${GRID},
    DATABASE   = ${DATABASE},
    STEP       = ${STEP},
    TARGET     = "/home/ecm5702/perm/reference/o96_o320/tc/fernanda/surface_pf_${STREAM}_${GRID}_${EXPID}_${my_date}.grib"
EOF
      mars my_request_${my_date}_${my_time}.mars
      status=$?

      sync
      sleep 5

      if [ $status -eq 0 ]; then
        rm -f my_request_${my_date}_${my_time}.mars
      fi
      done
    done
  done
done


## DORA

AREA="25/175/5/-105"
GRID="O320"
STREAM="ENFO"
TYPE="PF"
CLASS="RD"
STEP="24/to/120/by/24"
PARAMS="151.128/165.128/166.128/167.128"
NUMBER="1/to/10/by/1"
TIMES="0000"
YEAR="2023"
MONTH="08"
DATABASE="FDB"
LEVEL="SFC"
DAYS="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17"


for y in ${YEAR}; do
 
  for m in ${MONTH}; do
    #get the number of days for this particular month/year
    days_per_month=$(cal ${m} ${y} | awk 'NF {DAYS = $NF}; END {print DAYS}')
     
    #date loop
    for my_date in ${DAYS}; do
      my_date=${YEAR}${m}${my_date}
      echo $my_date
       
      #time lop
      for my_time in ${TIMES}; do
        cat << EOF > my_request_${my_date}_${my_time}.mars

RETRIEVE,
    CLASS      = ${CLASS},
    TYPE       = ${TYPE},
    STREAM     = ${STREAM},
    EXPVER     = ${EXPID},
    LEVTYPE    = ${LEVEL},
    AREA       = ${AREA},
    PARAM      = ${PARAMS},
    DATE       = ${my_date},
    TIME       = ${my_time},
    NUMBER     = ${NUMBER},
    GRID       = ${GRID},
    DATABASE   = ${DATABASE},
    STEP       = ${STEP},
    TARGET     = "/home/ecm5702/perm/reference/o96_o320/tc/dora/surface_pf_${STREAM}_${GRID}_${EXPID}_${my_date}.grib"
EOF
      mars my_request_${my_date}_${my_time}.mars
      status=$?

      sync
      sleep 5

      if [ $status -eq 0 ]; then
        rm -f my_request_${my_date}_${my_time}.mars
      fi
      done
    done
  done
done













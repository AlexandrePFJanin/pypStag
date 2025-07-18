Instructions for the 'stagyy-fields' file.

Pairing of stagyy field type identifier on the binary name and field names.

If you want to make modification or add new fields, enter them in the local file
"stagyy-fields-local" and do not modify the default file "stagyy-fields-defaults".
By default, pypStag will look into the local file. If the field is not found there
then, pypStag will look in the default file.


1. Basics pairing

    Format:
    identifier -> field name

    Example:
    eta  -> Viscosity


2. Specify an additionnal scalar field

    Where:
    Typicaly for the 'Velocity-Pressure' field (vp).
    The pressure field is an additional scalar field.
    
    Format:
    identifier -> field name -> +
    
    Example:
    vp   -> Velocity-Pressure -> +


3. Important

    The binary file containing the velocity field must
    have the string "Velocity" explicitly in its name
    (as in the default file) and must be the only one
    to contain "Velocity"


4. Apply the modification

    In order to apply the modifications on the pairing,
    you must link again (re-install) pypStag to your
    python. Follow the instructions on main pypStag
    README.md file.




